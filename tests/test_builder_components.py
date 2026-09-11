"""Offline component tests. No API access or benchmark examples required."""
import importlib
import json
import unittest
from pathlib import Path
from unittest.mock import patch

class BuilderTests(unittest.TestCase):
    def test_synthetic_query_parsing(self):
        task=json.loads((Path(__file__).resolve().parents[1]/'examples/synthetic_trace.json').read_text())[0]
        for adapter in ['standard','miro']:
            parser=importlib.import_module(f'searchatlas.builder.{adapter}.parsing')
            queries,_=parser.extract_queries(task['messages'])
            self.assertEqual([q['text'] for q in queries],['Example Hall opening year'])

    def test_fixed_model_response(self):
        response={'units':[{'unit_id':'u1','unit_type':'attribute','q0_span':'Example Hall'}]}
        for adapter in ['standard','miro']:
            constraint=importlib.import_module(f'searchatlas.builder.{adapter}.constraints')
            attribution=importlib.import_module(f'searchatlas.builder.{adapter}.attribution')
            with patch.object(attribution,'call_llm_json',return_value=response) as mock:
                result=constraint.decompose_q0_units('When did Example Hall open?')
            self.assertEqual(result,response['units']);self.assertEqual(mock.call_count,1)

    def test_parent_cover(self):
        edges=[{'source':q,'target':'q3','edge_kind':'evidence_derived','covered_signals':['hall']} for q in ['q1','q2']]
        for adapter in ['standard','miro']:
            module=importlib.import_module(f'searchatlas.builder.{adapter}.parent_selection')
            kept=module.apply_mpsc(edges,'q3',{'q1':{'hall'},'q2':{'hall'}},{'hall'},{'q1':1,'q2':2})
            self.assertEqual({e['source'] for e in kept},{'q2'})

if __name__=='__main__':unittest.main()
