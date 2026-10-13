"""Tests for the hybrid retrieval pipeline (no ML models needed).

Embeddings are a deterministic hashed bag-of-words; the reranker and QA reader
are stubbed. Run from backend/:  python -m pytest tests
"""

import os
import sys
import zlib

import numpy as np
import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import server  # noqa: E402


class FakeEmbeddings(Embeddings):
    DIM = 256

    def _embed(self, text):
        vec = np.zeros(self.DIM, dtype="float32")
        for tok in server._tokenize(text):
            vec[zlib.crc32(tok.encode()) % self.DIM] += 1.0
        norm = np.linalg.norm(vec)
        return (vec / norm if norm else vec).tolist()

    def embed_documents(self, texts):
        return [self._embed(t) for t in texts]

    def embed_query(self, text):
        return self._embed(text)


def vectors_for(chunks):
    """Exact in-memory vector backend over the (fake) embeddings of the chunk texts."""
    return server.NumpyVectors(FakeEmbeddings().embed_documents([c.page_content for c in chunks]))


def make_store(texts_by_page):
    """texts_by_page: list of (page_num, text). Mirrors _process_pdf_sync's setup."""
    chunks = [Document(page_content=t, metadata={"page": p - 1}) for p, t in texts_by_page]
    for i, c in enumerate(chunks):
        c.metadata["chunk_id"] = i
    index = server.MetadataIndex()
    for i, c in enumerate(chunks):
        index.add_chunk(server.ChunkMetadata(
            chunk_id=i, page_num=c.metadata["page"] + 1, section=f"Page {c.metadata['page'] + 1}",
            word_count=len(c.page_content.split()), tags=[], has_code=False,
            has_table=False, content_quality_score=0.8))
    index.finalize()
    store = server.DocumentStore(
        doc_id="d", vector_store=vectors_for(chunks),
        chunks=chunks, metadata_index=index, doc_info={}, bm25=server._build_bm25(chunks))
    return store


@pytest.fixture(autouse=True)
def fake_models(monkeypatch):
    monkeypatch.setattr(server.models, "embeddings", FakeEmbeddings())
    monkeypatch.setattr(server, "PERSIST", False)             # tests never write to the real data folder
    monkeypatch.setattr(server, "VECTOR_BACKEND", "numpy")    # Qdrant has its own tests below


def test_rrf_rewards_agreement_between_rankers():
    assert server._rrf_fuse([[1, 2, 3], [3, 1, 4]]) == [1, 3, 2, 4]


def test_rrf_single_ranking_keeps_order():
    assert server._rrf_fuse([[5, 9, 2]]) == [5, 9, 2]


def test_bm25_finds_exact_identifier():
    store = make_store([(1, "general overview of the platform"),
                        (2, "see clause QX7731 for termination terms"),
                        (3, "pricing and billing details")])
    assert server._bm25_search(store, "QX7731", None, 5) == [1]


def test_bm25_drops_chunks_with_no_term_match():
    store = make_store([(1, "alpha beta"), (2, "gamma delta")])
    assert server._bm25_search(store, "zzz", None, 5) == []


def test_dense_search_returns_chunk_ids_that_index_into_chunks():
    store = make_store([(1, "apples and oranges"), (2, "cars and trucks"), (3, "rivers and lakes")])
    top = server._dense_search(store, "cars trucks", None, 1)
    assert store.chunks[top[0]].page_content == "cars and trucks"


def test_duplicate_chunk_text_keeps_distinct_ids():
    # Old code matched chunks by text, so identical chunks all collapsed to the first one
    store = make_store([(1, "same text"), (2, "same text"), (3, "other")])
    ids = server._dense_search(store, "same text", None, 5)
    assert {0, 1} <= set(ids)


def test_prefilter_searches_only_chosen_pages_even_if_best_matches_are_elsewhere():
    # 40 strong matches on pages 1-40; only page 41-45 allowed and they match weakly.
    pages = [(p, "tax invoice payment terms") for p in range(1, 41)]
    pages += [(p, f"appendix note {p} with payment") for p in range(41, 46)]
    store = make_store(pages)
    allowed = store.metadata_index.pre_filter(page_start=41, page_end=45)
    assert len(allowed) == 5

    dense = server._dense_search(store, "tax invoice payment terms", allowed, server.DENSE_TOP_K)
    assert dense and set(dense) <= allowed          # old code returned nothing here
    sparse = server._bm25_search(store, "tax invoice payment terms", allowed, server.BM25_TOP_K)
    assert sparse and set(sparse) <= allowed


def _install_pipeline_stubs(monkeypatch, store, rerank_score):
    monkeypatch.setitem(server.document_stores, "d", store)
    monkeypatch.setattr(
        server.models, "rerank_documents",
        lambda q, docs, top_n=5: [(d, rerank_score - 0.01 * i) for i, d in enumerate(docs[:top_n])])
    monkeypatch.setattr(
        server.models, "qa_reader",
        lambda inputs: {"answer": "42 days", "score": 0.9}, raising=False)


def test_confidence_is_top_reranker_score(monkeypatch):
    store = make_store([(1, "the notice period is 42 days"), (2, "unrelated text here")])
    _install_pipeline_stubs(monkeypatch, store, rerank_score=0.83)
    out = server._ask_question_sync("d", "what is the notice period", "u",
                                    None, None, None, None, None, None)
    assert out["answer_found"] is True
    assert out["score"] == 0.83
    assert out["qa_score"] == 0.9
    assert out["answer"] == "42 days"
    assert out["method"] == "extractive_qa"


def test_low_reranker_score_returns_not_found_with_closest_passages(monkeypatch):
    store = make_store([(1, "the notice period is 42 days"), (2, "unrelated text here")])
    _install_pipeline_stubs(monkeypatch, store, rerank_score=0.02)
    out = server._ask_question_sync("d", "what is the notice period", "u",
                                    None, None, None, None, None, None)
    assert out["answer_found"] is False
    assert out["method"] == "not_found"
    assert out["score"] == 0.02
    assert "couldn't find" in out["answer"]
    assert out["source_chunks"]


def test_filter_matching_nothing_returns_uniform_empty_result(monkeypatch):
    store = make_store([(1, "alpha"), (2, "beta")])
    _install_pipeline_stubs(monkeypatch, store, rerank_score=0.9)
    out = server._ask_question_sync("d", "alpha", "u", 50, 60, None, None, None, None)
    assert out["answer_found"] is False
    assert out["pipeline_stages"]["stage1_chunks_after"] == 0   # UI reads these keys
    assert "stage2_search_ms" in out["pipeline_stages"]


def test_page_filter_end_to_end_only_returns_chunks_from_those_pages(monkeypatch):
    pages = [(p, "tax invoice payment terms") for p in range(1, 41)]
    pages += [(41, "payment appendix"), (42, "payment schedule")]
    store = make_store(pages)
    _install_pipeline_stubs(monkeypatch, store, rerank_score=0.8)
    out = server._ask_question_sync("d", "payment terms", "u", 41, 42, None, None, None, None)
    assert out["answer_found"] is True
    assert all("appendix" in c or "schedule" in c for c in out["source_chunks"])


# ---------------------------------------------------------------------------
# Structured answers
# ---------------------------------------------------------------------------

from langchain_text_splitters import RecursiveCharacterTextSplitter  # noqa: E402

PAGE_4 = (
    "Part 4: Self-Learning Models\n\n"
    "R-Zero evolves from zero data with a challenger and solver. "
    "SEAL lets a model generate its own weight updates. "
    "AlphaProof uses self-play in formal proof space. "
    "TT-SI applies temporary gradient updates during inference.\n\n"
    "Market Status\n\n"
    "No commercial API offers true test-time adaptation. All models are static during inference."
)
PAGE_7 = (
    "Part 7: Root Problems\n\n"
    "Recurrence gives fourteen effective layers for fixed depth. "
    "Working memory is mitigated because tools offload memory to storage."
)


def make_text_store(pages, chunk_size=180, overlap=40):
    """Real splitter with offsets, so chunks are cut mid-sentence like in a real PDF."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=overlap, add_start_index=True,
        separators=["\n\n", "\n", ". ", " ", ""])
    docs = [Document(page_content=t, metadata={"page": i}) for i, t in enumerate(pages)]
    chunks = splitter.split_documents(docs)
    index = server.MetadataIndex()
    for i, c in enumerate(chunks):
        c.metadata["chunk_id"] = i
        page = c.metadata["page"] + 1
        index.add_chunk(server.ChunkMetadata(
            chunk_id=i, page_num=page, section=server._detect_section(c.page_content, page),
            word_count=len(c.page_content.split()), tags=[], has_code=False,
            has_table=False, content_quality_score=0.8))
    index.finalize()
    store = server.DocumentStore(
        doc_id="d", vector_store=vectors_for(chunks), chunks=chunks, metadata_index=index, doc_info={},
        bm25=server._build_bm25(chunks), page_texts={i + 1: t for i, t in enumerate(pages)})
    return store


def test_clean_text_repairs_pdf_damage():
    raw = "The eﬀect of trans-\nformers on NTP �reasoning.\nWrapped line\ncontinues here."
    out = server._clean_text(raw)
    assert "effect" in out                      # ligature expanded
    assert "transformers" in out                # hyphenated break joined
    assert "�" not in out                  # unmappable glyph dropped
    assert "Wrapped line continues here." in out  # hard wrap re-joined


def test_clean_text_keeps_headings_and_bullets_on_their_own_lines():
    out = server._clean_text("Intro sentence.\nPart 4: Heading\n• item one\n• item two")
    assert out.split("\n") == ["Intro sentence.", "Part 4: Heading", "• item one", "• item two"]


def test_segment_page_offsets_point_back_into_the_text():
    text = "First sentence here. Second sentence follows.\n• A bullet line that is long enough"
    for start, end, seg in server._segment_page(text):
        assert seg in text[start:end] or seg == text[start:end].strip()


def _all_chunks_ranked(store, reverse=True):
    chunks = list(reversed(store.chunks)) if reverse else list(store.chunks)
    return [(c, 0.9 - 0.01 * i) for i, c in enumerate(chunks)]


def test_structured_answer_uses_whole_sentences_only():
    store = make_text_store([PAGE_4, PAGE_7])
    out = server._build_structured_answer("tell me about self-learning models", _all_chunks_ranked(store), store)
    items = [i for b in out["answer_blocks"] for i in b["items"]]
    assert items
    for item in items:
        assert item[0].isupper() or item[0].isdigit(), item   # never starts mid-sentence
        assert item.rstrip()[-1] in ".!?", item               # never cut off at the end


def test_structured_answer_has_no_duplicate_items_despite_chunk_overlap():
    store = make_text_store([PAGE_4, PAGE_7], overlap=80)
    out = server._build_structured_answer("self-learning models", _all_chunks_ranked(store), store)
    items = [i for b in out["answer_blocks"] for i in b["items"]]
    assert len(items) == len(set(items))


def test_structured_answer_is_in_document_order_not_relevance_order():
    store = make_text_store([PAGE_4, PAGE_7])
    # Rank page-7 chunks first: output must still list page 4 before page 7
    ranked = sorted(_all_chunks_ranked(store, reverse=False),
                    key=lambda r: -r[0].metadata["page"])
    out = server._build_structured_answer("models", ranked, store)
    pages = [b["page"] for b in out["answer_blocks"]]
    assert pages == sorted(pages) and set(pages) == {1, 2}


def test_structured_answer_groups_by_heading_and_labels_pages():
    store = make_text_store([PAGE_4, PAGE_7])
    out = server._build_structured_answer("self-learning models", _all_chunks_ranked(store), store)
    headings = {b["heading"] for b in out["answer_blocks"]}
    assert "Self-Learning Models" in headings or any("Self-Learning" in h for h in headings)
    assert "(p. 1)" in out["answer"] or "Page 1" in out["answer"]
    assert out["answer"].count("•") == sum(len(b["items"]) for b in out["answer_blocks"])


def test_structured_answer_prefers_sentences_matching_the_question():
    store = make_text_store([PAGE_4], chunk_size=1000)
    out = server._build_structured_answer("tell me about AlphaProof", _all_chunks_ranked(store), store)
    first = out["answer_blocks"][0]["items"]
    assert any("AlphaProof" in i for i in first)


def test_structured_answer_with_nothing_returns_none_method():
    store = make_text_store([PAGE_4])
    out = server._build_structured_answer("anything", [], store)
    assert out["method"] == "none" and out["answer_blocks"] == []


def test_broad_question_endpoint_returns_blocks_and_found_flag(monkeypatch):
    store = make_text_store([PAGE_4, PAGE_7])
    monkeypatch.setitem(server.document_stores, "d", store)
    monkeypatch.setattr(server.models, "rerank_documents",
                        lambda q, docs, top_n=5: [(d, 0.8 - 0.01 * i) for i, d in enumerate(docs[:top_n])])
    out = server._ask_question_sync("d", "Tell me about self-learning models", "u",
                                    None, None, None, None, None, None)
    assert out["answer_found"] is True
    assert out["method"] == "semantic_retrieval"
    assert out["answer_blocks"] and out["score"] == 0.8


def test_clean_text_maps_symbol_font_bullets():
    assert server._clean_text(" first point\n second point") == "• first point\n• second point"


def test_heading_line_with_abbreviation_stays_one_segment():
    text = "Part 4: Self-Learning LLMs — Research vs. Reality\n\nBody sentence follows here."
    segs = [s for _, _, s in server._segment_page(text)]
    assert segs[0] == "Part 4: Self-Learning LLMs — Research vs. Reality"


def test_abbreviation_inside_a_sentence_does_not_split_it():
    line = ("Use tools, e.g. Python or Rust, for this task when the problem needs exact computation steps. "
            "Next one follows here with more detail about it.")
    segs = [s for _, _, s in server._segment_page(line)]
    assert segs[0].startswith("Use tools, e.g. Python or Rust,") and segs[0].endswith("computation steps.")
    assert len(segs) == 2


def test_question_style_title_is_a_heading_but_a_short_question_is_not():
    assert server._is_heading("Part 7: Can Axiom-3B Solve the 5 Root Problems?")
    assert not server._is_heading("is this really true?")
    assert not server._is_heading("The model failed to converge on this.")


def test_weak_chunks_are_left_out_of_the_answer():
    store = make_text_store([PAGE_4, PAGE_7], chunk_size=1000)
    p4 = [c for c in store.chunks if c.metadata["page"] == 0]
    p7 = [c for c in store.chunks if c.metadata["page"] == 1]
    ranked = [(p4[0], 0.95), (p7[0], 0.05)]          # page 7 scores far below page 4
    out = server._build_structured_answer("self-learning models", ranked, store)
    assert {b["page"] for b in out["answer_blocks"]} == {1}


def test_identical_sentences_appear_once():
    page = ("Intro Section\n\nThe same sentence is repeated here. The same sentence is repeated here. "
            "Another point follows afterwards with enough extra words to pass the length limit.")
    store = make_text_store([page], chunk_size=1000)
    out = server._build_structured_answer("same sentence", [(store.chunks[0], 0.9)], store)
    items = [i for b in out["answer_blocks"] for i in b["items"]]
    assert items.count("The same sentence is repeated here.") == 1


def test_items_carry_no_leading_bullet_character():
    page = "Findings\n\n• First finding is long enough to keep.\n• Second finding is also long enough."
    store = make_text_store([page], chunk_size=1000)
    out = server._build_structured_answer("findings", [(store.chunks[0], 0.9)], store)
    assert all(not i.startswith("•") for b in out["answer_blocks"] for i in b["items"])


def test_list_items_and_short_lines_are_not_split_into_fragments():
    text = ("• SEAL (MIT): Model generates its own weight updates. Persistent changes via self-edits.\n"
            "2048 is still 2048. Tools offload memory.\n"
            "1. Fixed Depth: A transformer has a fixed number of layers. If a task needs 25 hops it fails.")
    segs = [s for _, _, s in server._segment_page(text)]
    assert segs[0].endswith("Persistent changes via self-edits.")
    assert segs[1] == "2048 is still 2048. Tools offload memory."
    assert segs[2].endswith("it fails.") and len(segs) == 3


def test_long_prose_lines_are_still_split_into_sentences():
    line = ("This first sentence is deliberately long enough to push the line past the length limit. "
            "This second sentence follows right after it. And a third one ends the line here.")
    assert len([s for _, _, s in server._segment_page(line)]) == 3


def test_relative_cutoff_keeps_chunks_when_all_scores_are_low():
    # A good broad answer can score ~0.12 overall; relative filtering must not empty it
    store = make_text_store([PAGE_4, PAGE_7], chunk_size=1000)
    p4 = [c for c in store.chunks if c.metadata["page"] == 0]
    out = server._build_structured_answer("self-learning models", [(p4[0], 0.125)], store)
    assert out["method"] == "semantic_retrieval" and out["answer_blocks"]


def test_numbered_sentence_is_not_a_section_heading_but_numbered_title_is():
    sentence = "4. Self-learning LLMs exist in research, not in product. No commercial API offers adaptation."
    assert server._detect_section(sentence + "\nmore text", 3) == "Page 3"
    assert server._detect_section("2.1 Research Methods\nbody text follows", 3) == "Research Methods"


def test_not_found_response_lists_closest_sections(monkeypatch):
    store = make_text_store([PAGE_4, PAGE_7])
    monkeypatch.setitem(server.document_stores, "d", store)
    monkeypatch.setattr(server.models, "rerank_documents",
                        lambda q, docs, top_n=5: [(d, 0.02 - 0.001 * i) for i, d in enumerate(docs[:top_n])])
    out = server._ask_question_sync("d", "explain the transformer architecture", "u",
                                    None, None, None, None, None, None)
    assert out["answer_found"] is False and out["method"] == "not_found"
    assert out["answer_blocks"], "closest sections should still be offered"
    assert all("page" in b and b["items"] for b in out["answer_blocks"])


def test_clean_text_removes_noncharacter_glyph_placeholders():
    # U+FFFF is what some PDF extractors emit for a glyph they cannot map (e.g. "≠")
    out = server._clean_text("Next-Token Prediction ￿Reasoning and ￾ more ﷐ text")
    assert out == "Next-Token Prediction Reasoning and more text"


# ---------------------------------------------------------------------------
# Tables, chunk context, candidate pool, broad answers
# ---------------------------------------------------------------------------

import pymupdf  # noqa: E402


def make_pdf(pages):
    """pages: list of [(x, y, text)] placed with insert_text -> one line per item."""
    doc = pymupdf.open()
    for items in pages:
        page = doc.new_page()
        for x, y, text in items:
            page.insert_text((x, y), text, fontsize=9)
    return doc.tobytes()


TABLE_PAGE = [
    (72, 60, "Part 8: 4-Domain Training Feasibility"),
    (72, 100, "Domain"), (150, 100, "Data"), (300, 100, "Expected Strength"),
    (72, 120, "DSA"), (150, 120, "Excellent"), (300, 120, "Strongest"),
    (72, 135, "ML Code"), (150, 135, "Excellent"), (300, 135, "Mixed (syntax yes,"),
    (300, 148, "convergence no)"),
    (72, 165, "Agents"), (150, 165, "Moderate"), (300, 165, "Good"),
    (72, 260, "This paragraph follows the table and is long enough to be read as prose, not as a cell."),
]


def page_text_of(items):
    doc = pymupdf.open(stream=make_pdf([items]), filetype="pdf")
    return server._page_text_with_tables(doc[0])


def test_borderless_table_becomes_labelled_rows():
    text = page_text_of(TABLE_PAGE)
    assert "• Domain: DSA; Data: Excellent; Expected Strength: Strongest" in text
    assert "• Domain: Agents; Data: Moderate; Expected Strength: Good" in text


def test_wrapped_table_cell_is_joined_into_its_row():
    text = page_text_of(TABLE_PAGE)
    assert "Expected Strength: Mixed (syntax yes, convergence no)" in text


def test_text_around_a_table_is_kept_in_order():
    lines = page_text_of(TABLE_PAGE).split("\n")
    assert lines[0] == "Part 8: 4-Domain Training Feasibility"
    assert lines[-1].startswith("This paragraph follows the table")
    assert len(lines) == 1 + 3 + 1  # heading + 3 rows + paragraph


def test_two_column_prose_is_not_turned_into_a_table():
    left = "The first column carries a long line of ordinary running prose text"
    right = "The second column also carries a long line of ordinary prose text"
    items = [(72, 80 + 14 * i, f"{left} {i}.") for i in range(8)] + \
            [(320, 80 + 14 * i, f"{right} {i}.") for i in range(8)]
    text = page_text_of(items)
    assert "• " not in text
    assert text.count("ordinary") == 16


def test_page_without_a_table_is_unchanged():
    items = [(72, 80 + 14 * i, f"Plain paragraph line number {i} of the document.") for i in range(8)]
    text = page_text_of(items)
    assert text.split("\n") == [f"Plain paragraph line number {i} of the document." for i in range(8)]


def test_chunk_contexts_carry_title_part_and_nearest_heading():
    pages = {1: "Session Summary: Axiom-3B Design\n\nPart 5: Architecture Design\n\nHyperparameters\n"
                "• Context: 16K\n• Layers: 18\n"}
    chunk = Document(page_content="• Context: 16K\n• Layers: 18\n",
                     metadata={"page": 0, "start_index": pages[1].index("• Context")})
    (ctx, heading), = server._chunk_contexts([chunk], pages, "Session Summary: Axiom-3B Design")
    assert ctx == "Session Summary: Axiom-3B Design > Part 5: Architecture Design > Hyperparameters"
    assert heading == "Hyperparameters"


def test_chunk_context_carries_over_from_the_previous_page():
    pages = {1: "Intro\n\nStage 1: Pretraining\n\nSome text here that is the end of the page.",
             2: "• Optimizer: Muon and AdamW are used here\n• Precision: BF16 is used here"}
    chunk = Document(page_content=pages[2], metadata={"page": 1, "start_index": 0})
    (ctx, heading), = server._chunk_contexts([chunk], pages, "Doc")
    assert heading == "Stage 1: Pretraining" and "Stage 1: Pretraining" in ctx


def test_candidate_pool_keeps_a_chunk_that_only_bm25_ranks_well():
    dense = list(range(100, 130))                                   # dense: ids 100..129
    sparse = list(range(200, 212)) + [7] + list(range(212, 230))    # id 7 is BM25's 13th
    assert 7 not in server._rrf_fuse([dense, sparse])[:server.FUSED_TOP_K]   # the old cut drops it
    pool = server._candidate_pool(dense, sparse)
    assert 7 in pool
    assert len(pool) <= server.CANDIDATE_CAP and len(pool) == len(set(pool))


def test_candidate_pool_never_exceeds_the_cap():
    pool = server._candidate_pool(list(range(0, 30)), list(range(1000, 1030)))
    assert len(pool) == server.CANDIDATE_CAP


LIST_PAGE = ("Part 3: The Five Root Causes\n\n"
             "1. Fixed Depth = Fixed Reasoning Budget: layers equal rounds of computation.\n"
             "2. Working Memory Collapse: the hidden dimension is physical working memory.\n"
             "3. Compositionality Breakdown: attention is a parallel pattern matcher.\n"
             "4. Width Cannot Substitute for Depth: there are provable depth-width tradeoffs.\n"
             "5. Next-Token Prediction: it optimises pattern completion, not deliberate computation.")


def test_broad_answers_list_every_item_of_a_list():
    store = make_text_store([LIST_PAGE], chunk_size=1000)
    ranked = [(store.chunks[0], 0.9)]
    narrow = server._build_structured_answer("summarize the five root causes", ranked, store)
    broad = server._build_structured_answer("summarize the five root causes", ranked, store, broad=True)
    count = lambda out: sum(len(b["items"]) for b in out["answer_blocks"])
    assert count(broad) == 5 and count(narrow) <= server.MAX_ITEMS_PER_CHUNK
    text = " ".join(i for b in broad["answer_blocks"] for i in b["items"])
    for key in ("Fixed Depth", "Working Memory Collapse", "Compositionality", "Width Cannot", "Next-Token"):
        assert key in text


@pytest.fixture
def real_splitter(monkeypatch):
    monkeypatch.setattr(server.models, "text_splitter", RecursiveCharacterTextSplitter(
        chunk_size=300, chunk_overlap=60, add_start_index=True,
        separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""]))


def _upload(items_by_page, name="doc.pdf"):
    return server._process_pdf_sync(make_pdf(items_by_page), name, "tester")


def _spec_page():
    """Title, Part and sub-heading, then 14 spec lines: later chunks no longer contain the headings."""
    items = [(72, 60, "Axiom Design Summary"), (72, 80, "Part 5: Architecture Design"), (72, 100, "Hyperparameters")]
    items += [(72, 120 + 13 * k, f"• Setting number {k}: value {k} for the model") for k in range(14)]
    items += [(72, 120 + 13 * 14, "• Layers: 18 (6P + 8R + 4C)")]
    return items


def test_upload_indexes_context_but_keeps_plain_chunks(real_splitter):
    out = _upload([_spec_page()])
    store = server.document_stores[out["doc_id"]]
    spec = next(i for i, c in enumerate(store.chunks) if "Layers: 18" in c.page_content)
    assert "Hyperparameters" not in store.chunks[spec].page_content     # this chunk lost its heading
    indexed = store.indexed_chunks[spec].page_content
    assert indexed.startswith("Axiom Design Summary > Part 5: Architecture Design > Hyperparameters\n")
    assert indexed.endswith(store.chunks[spec].page_content)            # raw text unchanged after the context
    assert store.chunks[spec].metadata["chunk_id"] == store.indexed_chunks[spec].metadata["chunk_id"] == spec


def test_context_makes_a_bare_spec_chunk_findable_by_heading_words(real_splitter):
    page2 = [(72, 60, "Other Section"), (72, 90, "Unrelated prose about meetings and contacts for the team.")]
    out = _upload([_spec_page(), page2])
    store = server.document_stores[out["doc_id"]]
    hits = server._bm25_search(store, "hyperparameters layers", None, 5)
    top = store.chunks[hits[0]].page_content
    assert "Layers: 18" in top and "Hyperparameters" not in top   # matched through the context line only


def test_ask_returns_plain_chunks_without_the_context_line(real_splitter, monkeypatch):
    page1 = [(72, 60, "Axiom Design Summary"), (72, 90, "Hyperparameters"),
             (72, 110, "• Context: 16K tokens of honest effective length"),
             (72, 124, "• Layers: 18 in total across prelude, core and coda")]
    out = _upload([page1])
    monkeypatch.setattr(server.models, "rerank_documents",
                        lambda q, docs, top_n=5: [(d, 0.9 - 0.01 * i) for i, d in enumerate(docs[:top_n])])
    monkeypatch.setattr(server.models, "qa_reader", lambda inputs: {"answer": "16K", "score": 0.9}, raising=False)
    res = server._ask_question_sync(out["doc_id"], "What is the context length?", "u",
                                    None, None, None, None, None, None)
    assert res["answer_found"] is True
    assert all("Axiom Design Summary >" not in c for c in res["source_chunks"])


def test_upload_falls_back_to_the_plain_loader_if_table_extraction_fails(real_splitter, monkeypatch):
    def boom(path):
        raise RuntimeError("layout analysis failed")
    monkeypatch.setattr(server, "_extract_pages", boom)
    out = _upload([[(72, 60, "Fallback Title"), (72, 90, "Some plain text that should still be indexed fine.")]])
    assert out["num_chunks"] >= 1


def test_repeated_footer_is_removed_but_varying_margin_text_is_kept():
    pages = []
    for n in range(4):
        pages.append([(72, 100, f"Body text of page {n} with enough words to be real content."),
                      (72, 800, "Generated by Kimi"),
                      (300, 820, f"Page {n + 1}")] + ([(72, 40, "Only on the first page")] if n == 0 else []))
    doc = pymupdf.open(stream=make_pdf(pages), filetype="pdf")
    skip = frozenset(server._repeated_margin_lines(doc))
    assert server._margin_key("Generated by Kimi") in skip
    assert server._margin_key("Page 7") in skip                       # page numbers repeat too
    assert server._margin_key("Only on the first page") not in skip
    text = server._page_text_with_tables(doc[1], skip)
    assert "Generated by Kimi" not in text and "Body text of page 1" in text
    assert "Only on the first page" in server._page_text_with_tables(doc[0], skip)


def test_short_documents_never_lose_margin_text():
    doc = pymupdf.open(stream=make_pdf([[(72, 800, "Generated by Kimi"), (72, 100, "Body line of page one here.")],
                                        [(72, 800, "Generated by Kimi"), (72, 100, "Body line of page two here.")]]),
                       filetype="pdf")
    assert server._repeated_margin_lines(doc) == set()   # needs 3+ pages before anything counts as a footer


def _spy_store(monkeypatch, score):
    """Store whose chunks carry a context line; the reranker stub records what text it was given."""
    store = make_text_store([PAGE_4, PAGE_7])
    store.indexed_chunks = [Document(page_content="CTX LINE\n" + c.page_content, metadata=dict(c.metadata))
                            for c in store.chunks]
    seen = {}

    def fake_rerank(q, docs, top_n=5):
        seen["texts"] = [d.page_content for d in docs]
        return [(d, score - 0.001 * i) for i, d in enumerate(docs[:top_n])]

    monkeypatch.setitem(server.document_stores, "d", store)
    monkeypatch.setattr(server.models, "rerank_documents", fake_rerank)
    monkeypatch.setattr(server.models, "qa_reader", lambda inputs: {"answer": "x", "score": 0.9}, raising=False)
    return seen


def test_factual_questions_are_reranked_on_plain_text_with_the_strict_gate(monkeypatch):
    seen = _spy_store(monkeypatch, score=0.05)
    out = server._ask_question_sync("d", "which tool is used here", "u", None, None, None, None, None, None)
    assert all(not t.startswith("CTX LINE") for t in seen["texts"])
    assert out["answer_found"] is False and out["method"] == "not_found"      # 0.05 < 0.10


def test_broad_questions_see_the_context_line_and_use_the_lower_gate(monkeypatch):
    seen = _spy_store(monkeypatch, score=0.05)
    out = server._ask_question_sync("d", "explain the self-learning models", "u", None, None, None, None, None, None)
    assert all(t.startswith("CTX LINE") for t in seen["texts"])
    assert out["answer_found"] is True and out["method"] == "semantic_retrieval"   # 0.05 >= 0.03
    assert all("CTX LINE" not in c for c in out["source_chunks"])                 # users never see it


# ---------------------------------------------------------------------------
# Font-aware headings, margin text, numeric tables (found on a second, real document)
# ---------------------------------------------------------------------------

import tempfile  # noqa: E402


def make_styled_pdf(pages):
    """pages: list of [(x, y, text, size, bold)] — real font sizes/weights, one line per item."""
    doc = pymupdf.open()
    for items in pages:
        page = doc.new_page()
        for x, y, text, size, bold in items:
            page.insert_text((x, y), text, fontsize=size, fontname="hebo" if bold else "helv")
    return doc.tobytes()


def extract_pdf(pdf_bytes):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(pdf_bytes)
    try:
        return server._extract_pages(tmp.name)
    finally:
        os.unlink(tmp.name)


def _body(n, y0=200):
    return [(72, y0 + 14 * i, f"Body sentence number {i} written in ordinary running text.", 9, False)
            for i in range(n)]


def test_typographic_headings_are_marked_and_body_text_is_not():
    page = ([(72, 60, "Large Document Title", 18, True), (72, 100, "2 Pretraining", 13, True)]
            + _body(6, 130) + [(72, 260, "2.1 Pretraining Data", 11, True)] + _body(4, 285)
            + [(72, 400, "3 Fine-tuning", 13, True)] + _body(3, 425))
    pages = extract_pdf(make_styled_pdf([page]))
    lines = pages[0].page_content.split("\n")
    marked = [ln[3:] for ln in lines if ln.startswith("## ")]
    assert marked == ["Large Document Title", "2 Pretraining", "2.1 Pretraining Data", "3 Fine-tuning"]
    assert not any(ln.startswith("## Body") for ln in lines)


def test_a_document_with_fewer_than_three_headings_gets_no_markers():
    page = [(72, 60, "Only Heading", 14, True)] + _body(8, 100)
    pages = extract_pdf(make_styled_pdf([page]))
    assert "## " not in pages[0].page_content


def test_marked_documents_ignore_the_text_heuristic():
    assert server._is_heading("Part 4: Self-Learning Models")                      # unmarked docs: heuristic
    assert not server._is_heading("Part 4: Self-Learning Models", marked=True)     # marked docs: fonts decide
    assert server._is_heading("## Part 4: Self-Learning Models", marked=True)


def test_marked_section_detection_and_context_use_clean_labels():
    pages = {1: "## Llama 2 Paper\n## 3 Fine-tuning\n## 3.1 Supervised Fine-Tuning\nWe collected annotations here.\n"}
    chunk = Document(page_content="We collected annotations here.\n",
                     metadata={"page": 0, "start_index": pages[1].index("We collected")})
    (ctx, heading), = server._chunk_contexts([chunk], pages, "Llama 2 Paper", marked=True)
    assert heading == "3.1 Supervised Fine-Tuning"
    assert ctx == "Llama 2 Paper > 3 Fine-tuning > 3.1 Supervised Fine-Tuning"       # no '##' leaks in
    assert server._detect_section("## 3.1 Supervised Fine-Tuning\nbody", 5, marked=True) == "3.1 Supervised Fine-Tuning"
    assert server._detect_section("Some Body Words Here\nmore text", 5, marked=True) == "Page 5"


def test_visible_text_has_no_heading_markers():
    assert server._visible("## 2 Pretraining\nbody text") == "2 Pretraining\nbody text"


def test_bare_numbers_in_a_table_are_not_dropped_as_page_numbers():
    pages = []
    for n in range(4):
        items = [(72, 100, f"Paragraph on page {n} with ordinary words in it.", 9, False),
                 (300, 820, str(n + 1), 9, False)]                       # page number in the bottom margin
        if n == 1:   # a table whose cells are bare numbers, in the page body
            items += [(72, 200, "Model", 9, False), (200, 200, "GPU hours", 9, False), (330, 200, "Power", 9, False),
                      (72, 215, "Seven", 9, False), (200, 215, "184320", 9, False), (330, 215, "400", 9, False),
                      (72, 230, "Seventy", 9, False), (200, 230, "1720320", 9, False), (330, 230, "350", 9, False)]
        pages.append(items)
    text = extract_pdf(make_styled_pdf(pages))[1].page_content
    assert "184320" in text and "1720320" in text and "400" in text
    assert "GPU hours: 1720320" in text                                     # and the table was rebuilt
    assert not any(ln.strip() in ("1", "2", "3", "4") for ln in text.split("\n"))   # page numbers removed


def test_a_numeric_first_row_is_not_mistaken_for_a_table_header():
    items = [(72, 100, "7B", 9, False), (200, 100, "184320", 9, False), (330, 100, "400", 9, False),
             (72, 115, "13B", 9, False), (200, 115, "368640", 9, False), (330, 115, "400", 9, False),
             (72, 130, "34B", 9, False), (200, 130, "1038336", 9, False), (330, 130, "350", 9, False)]
    doc = pymupdf.open(stream=make_styled_pdf([items]), filetype="pdf")
    text = server._page_text_with_tables(doc[0])
    assert "• " not in text                                    # no invented labels like "7B: 13B"
    assert all(v in text for v in ("184320", "368640", "1038336"))   # and nothing lost


# ---------------------------------------------------------------------------
# Two-stage (section) search
# ---------------------------------------------------------------------------

def make_sectioned_store(n_sections=8, per_section=3, marked=True):
    """A document of `n_sections` headed sections; each has its own distinctive vocabulary."""
    topics = ["pretraining", "finetuning", "safety", "evaluation", "carbon", "tokenizer", "attention", "reward"]
    parts = []
    for s in range(n_sections):
        word = topics[s % len(topics)] + str(s)
        body = " ".join(f"The {word} section explains {word} details number {k} in plain running prose here." for k in range(per_section * 3))
        parts.append(f"## {s + 1} {word.title()}\n{body}")
    page_text = "\n".join(parts)
    chunks = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=0, add_start_index=True,
                                            separators=["\n\n", "\n", ". ", " ", ""]).split_documents(
        [Document(page_content=page_text, metadata={"page": 0})])
    for i, c in enumerate(chunks):
        c.metadata["chunk_id"] = i
    index = server.MetadataIndex()
    for i, c in enumerate(chunks):
        index.add_chunk(server.ChunkMetadata(chunk_id=i, page_num=1, section="s", word_count=10, tags=[],
                                             has_code=False, has_table=False, content_quality_score=0.8))
    index.finalize()
    store = server.DocumentStore(
        doc_id="d", vector_store=vectors_for(chunks), chunks=chunks,
        metadata_index=index, doc_info={}, bm25=server._build_bm25(chunks), page_texts={1: page_text}, marked=marked,
        section_index=server._build_section_index(chunks, {1: page_text}, "Doc", marked))
    return store


def test_sections_follow_headings_and_cover_every_chunk_once():
    store = make_sectioned_store()
    secs = store.section_index.sections
    assert len(secs) >= 8
    assert any("Pretraining0" in s["title"] for s in secs)
    all_ids = sorted(i for s in secs for i in s["chunk_ids"])
    assert all_ids == list(range(len(store.chunks)))                 # every chunk, exactly once
    assert max(len(s["chunk_ids"]) for s in secs) <= server.SECTION_MAX_CHUNKS


def test_section_scope_picks_the_section_that_matches_the_question():
    store = make_sectioned_store()
    vec = FakeEmbeddings().embed_query("tell me about carbon4 details")
    scope = server._section_scope(store.section_index, "tell me about carbon4 details", vec)
    texts = " ".join(store.chunks[i].page_content for i in scope)
    assert "carbon4" in texts
    assert len(scope) < len(store.chunks)                             # genuinely narrower than the whole document


def test_documents_without_headings_still_get_bounded_sections():
    store = make_sectioned_store(marked=False)                        # heuristic mode, headings are '## ' text
    assert all(len(s["chunk_ids"]) <= server.SECTION_MAX_CHUNKS for s in store.section_index.sections)


def _ranked_stub(monkeypatch, seen):
    def fake(q, docs, top_n=5):
        seen["ids"] = [d.metadata["chunk_id"] for d in docs]
        return [(d, 0.9 - 0.01 * i) for i, d in enumerate(docs[:top_n])]
    monkeypatch.setattr(server.models, "rerank_documents", fake)
    monkeypatch.setattr(server.models, "qa_reader", lambda inputs: {"answer": "x", "score": 0.9}, raising=False)


def test_two_stage_search_reaches_the_reranker_with_the_right_section(monkeypatch):
    store = make_sectioned_store()
    monkeypatch.setitem(server.document_stores, "d", store)
    seen = {}
    _ranked_stub(monkeypatch, seen)
    out = server._ask_question_sync("d", "carbon4 details", "u", None, None, None, None, None, None)
    assert out["pipeline_stages"].get("stage2_scope_chunks")
    assert any("carbon4" in store.chunks[i].page_content for i in seen["ids"])


def test_two_stage_can_be_switched_off(monkeypatch):
    store = make_sectioned_store()
    monkeypatch.setitem(server.document_stores, "d", store)
    monkeypatch.setattr(server, "TWO_STAGE", False)
    _ranked_stub(monkeypatch, {})
    out = server._ask_question_sync("d", "carbon4 details", "u", None, None, None, None, None, None)
    assert "stage2_scope_chunks" not in out["pipeline_stages"]


def test_small_documents_skip_the_section_stage(real_splitter):
    out = _upload([_spy_page_for_small_doc()])
    assert server.document_stores[out["doc_id"]].section_index is None   # far below TWO_STAGE_MIN_CHUNKS


def _spy_page_for_small_doc():
    return [(72, 60, "Tiny Document"), (72, 90, "Some plain text that is far too short to need sections at all.")]


def test_user_page_filters_still_apply_inside_two_stage_search(monkeypatch):
    store = make_sectioned_store()
    monkeypatch.setitem(server.document_stores, "d", store)
    seen = {}
    _ranked_stub(monkeypatch, seen)
    server._ask_question_sync("d", "carbon4 details", "u", 1, 1, None, None, None, None)   # page 1 is everything
    assert seen["ids"]


# ---------------------------------------------------------------------------
# Qdrant backend, persistence across restarts, per-user ownership
# ---------------------------------------------------------------------------

import asyncio  # noqa: E402
from fastapi import HTTPException  # noqa: E402


def _unit_vectors(n, dim, seed):
    rng = np.random.default_rng(seed)
    v = rng.normal(size=(n, dim)).astype("float32")
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def test_qdrant_search_matches_exact_search_with_and_without_filters():
    from qdrant_client import QdrantClient
    vecs = _unit_vectors(60, 16, 1)
    q = QdrantClient(":memory:")
    qd = server.QdrantVectors.create(q, "docA", "alice", vecs)
    exact = server.NumpyVectors(vecs)
    for seed in range(5):
        query = _unit_vectors(1, 16, 100 + seed)[0]
        assert [i for i, _ in qd.search(query, 10)] == [i for i, _ in exact.search(query, 10)]
        allowed = set(range(0, 60, 3))
        got = [i for i, _ in qd.search(query, 10, allowed)]
        assert got == [i for i, _ in exact.search(query, 10, allowed)] and set(got) <= allowed


def test_qdrant_keeps_documents_apart_in_one_collection():
    from qdrant_client import QdrantClient
    q = QdrantClient(":memory:")
    a, b = _unit_vectors(20, 8, 2), _unit_vectors(20, 8, 3)
    qa = server.QdrantVectors.create(q, "docA", "alice", a)
    qb = server.QdrantVectors.create(q, "docB", "bob", b)
    query = _unit_vectors(1, 8, 9)[0]
    assert [i for i, _ in qa.search(query, 5)] == [i for i, _ in server.NumpyVectors(a).search(query, 5)]
    assert [i for i, _ in qb.search(query, 5)] == [i for i, _ in server.NumpyVectors(b).search(query, 5)]


def test_numpy_search_with_an_empty_allowed_set_returns_nothing():
    assert server.NumpyVectors(_unit_vectors(5, 4, 0)).search(_unit_vectors(1, 4, 1)[0], 3, set()) == []


def test_qdrant_failure_falls_back_to_exact_in_memory_search(monkeypatch):
    monkeypatch.setattr(server, "VECTOR_BACKEND", "qdrant")
    monkeypatch.setattr(server, "_qdrant_client", lambda: (_ for _ in ()).throw(RuntimeError("locked")))
    backend = server._make_vector_backend("d", "u", _unit_vectors(4, 4, 0))
    assert isinstance(backend, server.NumpyVectors)


@pytest.fixture
def persist_env(tmp_path, monkeypatch):
    monkeypatch.setattr(server, "DATA_DIR", str(tmp_path))
    monkeypatch.setattr(server, "PERSIST", True)
    server._close_qdrant()
    yield tmp_path
    server._close_qdrant()
    server.document_stores.clear()
    server.doc_metadata.clear()


def _stub_models(monkeypatch):
    monkeypatch.setattr(server.models, "rerank_documents",
                        lambda q, docs, top_n=5: [(d, 0.9 - 0.01 * i) for i, d in enumerate(docs[:top_n])])
    monkeypatch.setattr(server.models, "qa_reader", lambda inputs: {"answer": "16K", "score": 0.9}, raising=False)


SPEC_PAGE = [(72, 60, "Axiom Design Summary"), (72, 90, "Hyperparameters"),
             (72, 110, "• Context: 16K tokens of honest effective length"),
             (72, 124, "• Layers: 18 in total across prelude, core and coda")]


@pytest.mark.parametrize("backend", ["qdrant", "numpy"])
def test_documents_survive_a_restart(persist_env, real_splitter, monkeypatch, backend):
    monkeypatch.setattr(server, "VECTOR_BACKEND", backend)
    _stub_models(monkeypatch)
    out = _upload([SPEC_PAGE])
    doc_id = out["doc_id"]
    before = server._ask_question_sync(doc_id, "What is the context length?", "tester", None, None, None, None, None, None)
    chunks_before = [c.page_content for c in server.document_stores[doc_id].chunks]

    server.document_stores.clear()          # simulate a restart: memory gone, only the data folder remains
    server.doc_metadata.clear()
    server._close_qdrant()
    assert server._load_documents() == 1

    store = server.document_stores[doc_id]
    assert [c.page_content for c in store.chunks] == chunks_before
    assert store.doc_info["uploaded_by"] == "tester" and server.doc_metadata[doc_id]["filename"] == "doc.pdf"
    after = server._ask_question_sync(doc_id, "What is the context length?", "tester", None, None, None, None, None, None)
    assert after["answer_found"] == before["answer_found"] and after["answer"] == before["answer"]
    assert after["source_chunks"] == before["source_chunks"]


def test_restored_large_documents_keep_their_section_index(persist_env, real_splitter, monkeypatch):
    monkeypatch.setattr(server, "VECTOR_BACKEND", "numpy")
    monkeypatch.setattr(server, "TWO_STAGE_MIN_CHUNKS", 1)
    out = _upload([_spec_page()])
    n_sections = len(server.document_stores[out["doc_id"]].section_index.sections)
    server.document_stores.clear()
    server._load_documents()
    index = server.document_stores[out["doc_id"]].section_index
    assert index is not None and len(index.sections) == n_sections and index.vectors.shape[0] == n_sections


def test_a_half_written_document_does_not_stop_the_others_from_loading(persist_env, real_splitter, monkeypatch):
    monkeypatch.setattr(server, "VECTOR_BACKEND", "numpy")
    good = _upload([SPEC_PAGE])["doc_id"]
    os.makedirs(os.path.join(server.DATA_DIR, "docs", "broken"), exist_ok=True)
    with open(os.path.join(server.DATA_DIR, "docs", "broken", "doc.json"), "w") as f:
        f.write('{"truncated": ')
    server.document_stores.clear()
    assert server._load_documents() == 1 and good in server.document_stores


def test_registered_users_survive_a_restart(persist_env):
    server.USERS_DB["zed"] = {"username": "zed", "full_name": "Zed", "hashed_password": "x"}
    try:
        server._save_users()
        del server.USERS_DB["zed"]
        assert server._load_users() == 1 and "zed" in server.USERS_DB
        saved = open(os.path.join(server.DATA_DIR, server.USERS_FILE_NAME), encoding="utf-8").read()
        assert "admin" not in saved                # built-in accounts are never written to disk
    finally:
        server.USERS_DB.pop("zed", None)


def test_nothing_is_written_when_persistence_is_off(tmp_path, monkeypatch, real_splitter):
    monkeypatch.setattr(server, "DATA_DIR", str(tmp_path))
    monkeypatch.setattr(server, "PERSIST", False)
    _upload([SPEC_PAGE])
    assert os.listdir(tmp_path) == []


def test_other_users_cannot_read_ask_or_list_a_document(real_splitter, monkeypatch):
    out = server._process_pdf_sync(make_pdf([SPEC_PAGE]), "alice.pdf", "alice")
    doc_id = out["doc_id"]
    alice, bob = {"username": "alice"}, {"username": "bob"}
    assert server._owned_store(doc_id, alice).doc_id == doc_id
    with pytest.raises(HTTPException) as e:
        server._owned_store(doc_id, bob)
    assert e.value.status_code == 404                                   # same answer as a missing document
    with pytest.raises(HTTPException) as e:
        asyncio.run(server.get_doc_info(doc_id, bob))
    assert e.value.status_code == 404
    monkeypatch.setattr(server.models, "_loaded", True)
    with pytest.raises(HTTPException) as e:
        asyncio.run(server.ask_question(server.AskRequest(doc_id=doc_id, question="What is the context length?"), bob))
    assert e.value.status_code == 404
    assert doc_id in asyncio.run(server.stats(alice))["documents"]
    assert doc_id not in asyncio.run(server.stats(bob))["documents"]    # filenames of others stay private
