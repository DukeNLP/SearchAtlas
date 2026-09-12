"""Citation-format regressions with synthetic evidence and fixed model outputs."""
from contextlib import ExitStack, redirect_stderr, redirect_stdout
from copy import deepcopy
import importlib
import io
import unittest
from unittest.mock import patch

from searchatlas.builder.citations import (
    CITATION_FORMAT_INSTRUCTIONS, normalize_provenance_citations,
)


def provenance(source="q1", kind="snippet", lo=0, hi=6):
    return {
        "source_qid": source, "source_turn": 1, "source_type": kind,
        "chunk_label": "[Tool_Response #2]", "window_lo": lo, "window_hi": hi,
        "window_text": "Orion Nebula is a fictional example here.",
        "hit_sentence_idx": lo, "hit_sentence": "Orion Nebula.",
    }


def payload():
    return {"candidates_by_token": {
        token: [{"source_qid": "q1", "provenance": [provenance()]}]
        for token in ("orion", "nebula")
    }}


class CitationTests(unittest.TestCase):
    rich = "[q1 snippet Tool_Response #2 window 0..6] supplies Orion Nebula."
    canonical = "[snippet] source=q1; chunk=Tool_Response #2; window=0..6 supplies Orion Nebula."

    def normalize(self, text, data=None, covered=None):
        return normalize_provenance_citations(
            text, "q1", payload() if data is None else data,
            {"orion", "nebula"} if covered is None else covered,
        )

    def test_both_adapters_require_the_same_prompt_format(self):
        for adapter in ("standard", "miro"):
            settings = importlib.import_module(f"searchatlas.builder.{adapter}.settings")
            self.assertIn(CITATION_FORMAT_INSTRUCTIONS, settings.PHASE1_SYSTEM_HYBRID)

    def test_checked_combined_reference_is_normalized(self):
        data = payload()
        before = deepcopy(data)
        self.assertEqual(self.normalize(self.rich, data), self.canonical)
        self.assertEqual(data, before)

    def test_canonical_reference_is_unchanged(self):
        self.assertEqual(self.normalize(self.canonical), self.canonical)

    def test_visit_requires_actual_visit_evidence(self):
        text = self.rich.replace("snippet", "visit")
        self.assertEqual(self.normalize(text), text)
        data = payload()
        for candidates in data["candidates_by_token"].values():
            candidates[0]["provenance"][0]["source_type"] = "visit"
        self.assertEqual(self.normalize(text, data), self.canonical.replace("snippet", "visit"))

    def test_unknown_source_chunk_and_window_are_not_repaired(self):
        for text in (self.rich.replace("q1 ", "q9 "),
                     self.rich.replace("#2", "#999"),
                     self.rich.replace("0..6", "0..999"),
                     self.rich.replace("0..6", "6..0")):
            with self.subTest(text=text):
                self.assertEqual(self.normalize(text), text)

    def test_missing_or_unrelated_coverage_is_not_repaired(self):
        for data, covered in (({}, {"orion"}), (payload(), set()), (payload(), {"lyra"})):
            self.assertEqual(self.normalize(self.rich, data, covered), self.rich)

    def test_plain_word_or_missing_position_does_not_invent_a_reference(self):
        for text in ("q1 snippet supplies Orion Nebula", "[q1 snippet] supplies Orion Nebula"):
            self.assertEqual(self.normalize(text), text)

    def test_all_references_must_be_checkable(self):
        text = self.rich + " Also " + self.rich.replace("#2", "#999")
        self.assertEqual(self.normalize(text), text)
        self.assertEqual(self.normalize(self.rich + " Also " + self.rich),
                         self.canonical + " Also " + self.canonical)

    def test_prompt_merged_window_is_recognized_but_gaps_are_not(self):
        data = payload()
        data["candidates_by_token"]["orion"][0]["provenance"] = [provenance(lo=0, hi=3)]
        data["candidates_by_token"]["nebula"][0]["provenance"] = [provenance(lo=3, hi=6)]
        self.assertEqual(self.normalize(self.rich, data), self.canonical)
        data["candidates_by_token"]["nebula"][0]["provenance"][0]["window_lo"] = 5
        self.assertEqual(self.normalize(self.rich, data), self.rich)


class AttributionCitationTests(unittest.TestCase):
    def classify(self, adapter, evidence_text, source="q1", edge_kind="evidence_derived", support=True):
        attribution = importlib.import_module(f"searchatlas.builder.{adapter}.attribution")
        evidence = importlib.import_module(f"searchatlas.builder.{adapter}.evidence")
        queries = [{"id": "q1", "text": "find object", "turn": 1},
                   {"id": "q2", "text": "Orion Nebula distance", "turn": 2}]
        empty = {"docs": {}, "result_counts": {}, "search_docs": {}, "visit_failures": {}, "urls": {}}
        kwargs = dict(question="find object", messages=[], queries=queries,
                      turn_to_qids={1: ["q1"], 2: ["q2"]}, top_k=5, sent_window=5,
                      max_snips=2, result_counts={}, qid_search_docs={}, turn_docs={},
                      visit_summary_hard_failures={}, query_url_metadata={},
                      turn_thinks={}, turn_contexts={1: deepcopy(empty), 2: deepcopy(empty)})
        if adapter == "miro":
            kwargs.update(turn_doc_chunks={}, turn_visible_tool_message_indices={1: set(), 2: {1}},
                          keep_tool_result=-1)
        raw = {"source": source, "target": "q2", "edge_kind": edge_kind,
               "evidence": evidence_text, "covered_signals": ["orion", "nebula"],
               "confidence": "high"}

        def hits(**params):
            token = params["token"]
            return [dict(provenance(), token=token)] if support and token in {"orion", "nebula"} else []

        with ExitStack() as stack:
            stack.enter_context(redirect_stdout(io.StringIO()))
            stack.enter_context(redirect_stderr(io.StringIO()))
            stack.enter_context(patch.object(evidence, "token_provenance_in_prior_docs", side_effect=hits))
            model = stack.enter_context(patch.object(attribution, "call_llm_json", side_effect=[
                {"edges": [], "no_source_found": ["q1"]},
                {"edges": [deepcopy(raw)], "no_source_found": []},
            ]))
            edges, turns, _ = attribution.phase1_hybrid_classify(**kwargs)
        self.assertEqual(model.call_count, 2, "Formatting must not add an API call")
        self.assertEqual(turns[-1]["raw_edges"], [raw], "Preserve the original model output")
        return edges, turns[-1]

    def test_both_formats_pass_validation_and_parent_selection(self):
        for adapter in ("standard", "miro"):
            for text in (CitationTests.rich, CitationTests.canonical):
                with self.subTest(adapter=adapter, evidence=text):
                    edges, turn = self.classify(adapter, text)
                    edge = next(e for e in edges if e["source"] == "q1" and e["target"] == "q2")
                    self.assertEqual(edge["evidence"], CitationTests.canonical)
                    self.assertEqual(set(edge["covered_signals"]), {"orion", "nebula"})
                    self.assertFalse(turn["rejected_edges"])
                    self.assertEqual(len(turn.get("citation_format_repairs", [])), int(text == CitationTests.rich))

    def test_bad_reference_still_rejected(self):
        for adapter in ("standard", "miro"):
            for text in (CitationTests.rich.replace("#2", "#999"),
                         CitationTests.rich.replace("0..6", "0..999"),
                         CitationTests.rich.replace("snippet", "visit")):
                edges, turn = self.classify(adapter, text)
                self.assertFalse(edges)
                self.assertEqual(turn["rejected_edges"][0]["reason"],
                                 "query_edge_missing_sentence_or_provenance_anchor")
                self.assertFalse(turn.get("citation_format_repairs"))

    def test_no_support_cannot_be_repaired(self):
        for adapter in ("standard", "miro"):
            edges, turn = self.classify(adapter, CitationTests.rich, support=False)
            self.assertFalse(edges)
            self.assertEqual(turn["rejected_edges"][0]["reason"], "source_has_no_admissible_provenance_coverage")

    def test_temporal_pk_and_failure_gates_are_unchanged(self):
        for adapter in ("standard", "miro"):
            for source, kind in (("q2", "evidence_derived"), ("q9", "evidence_derived"),
                                 ("Prior_knowledge", "prior_knowledge_derived"), ("q1", "failure_derived")):
                edges, turn = self.classify(adapter, CitationTests.rich, source=source, edge_kind=kind)
                self.assertFalse(edges)
                self.assertFalse(turn.get("citation_format_repairs"))


if __name__ == "__main__":
    unittest.main()
