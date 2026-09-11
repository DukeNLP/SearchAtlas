"""Optional lexical constraint matching when query-to-unit memberships are absent."""

import math
import re

from .graphs import is_query, validate_graph


STOPWORDS = set('''a an the is are was were be been being have has had do does did
will would can could may might must to of in for on with at by from as into through
during before after above below between out off over under again further then once
here there when where why how all each every both few more most other some such no
nor not only own same so than too very just what which who whom this that these
those it its and or but i me my we our you your he she they them their his her him'''.split())
GENERIC = set('''answer question provide please give tell name full exact exactly
looking want based detail details clue clues criteria identify find seeking as of
by before after between inclusive exclusive year years sometime somewhere around
prior later same person individual thing one two three four five six seven eight
nine ten this that asked written mentioned according available information'''.split())
TOKEN_STOPWORDS = set('''a an and are as at by for from in into is it of on or the
their to under was were what when which who with'''.split())


def _singularize(word):
    if len(word) <= 3:
        return word
    if word.endswith('ies') and len(word) > 4:
        return word[:-3] + 'y'
    if word.endswith(('ches', 'shes', 'xes', 'zes', 'sses')) and len(word) > 4:
        return word[:-2]
    return word[:-1] if word.endswith('s') and not word.endswith('ss') else word


def _distinctive_tokens(text):
    tokens = {_singularize(w) for w in re.findall(r'[a-z0-9]+', text.lower())
              if len(w) >= 2 or w.isdigit()}
    return tokens - STOPWORDS - GENERIC


def unit_matches(unit_text, query_text, rule):
    if rule == 'distinctive_terms':
        units, query = _distinctive_tokens(unit_text), _distinctive_tokens(query_text)
        hits = units & query
        if not hits:
            return False
        nonnumeric = {w for w in hits if not w.isdigit() and len(w) > 3}
        unit_nonnumeric = {w for w in units if not w.isdigit() and len(w) > 3}
        if len(units) == 1:
            return True
        if len(hits) >= 2 and (nonnumeric or not unit_nonnumeric):
            return True
        return len(hits) == 1 and bool(nonnumeric) and len(next(iter(hits))) >= 8
    if rule == 'token_coverage':
        drop = TOKEN_STOPWORDS | {'answer', 'final', 'question', 'source', 'evidence', 'specific', 'major'}
        def tokens(text):
            return {w for w in re.findall(r'[a-z0-9]+', text.lower()) if len(w) > 1} - drop
        units, query = tokens(unit_text), tokens(query_text)
        needed = 1 if len(units) <= 2 else max(2, math.ceil(0.5 * len(units)))
        return bool(units) and len(units & query) >= needed
    raise ValueError('Matching rule must be distinctive_terms or token_coverage')


def prepare_annotation(annotation, payload, rule=None):
    if annotation is None:
        return None
    unknown = set(annotation) - {'case_id', 'task_id', 'constraint_units', 'query_unit_ids'}
    if unknown:
        raise ValueError('Constraint annotations accept only units and query memberships')
    if 'query_unit_ids' in annotation:
        return annotation
    if rule is None:
        raise ValueError('Provide query_unit_ids or select --match-rule explicitly')
    units = annotation.get('constraint_units')
    if not isinstance(units, list):
        raise ValueError('constraint_units must be a list')
    for unit in units:
        if not isinstance(unit, dict) or not isinstance(unit.get('text'), str) or not unit['text'].strip():
            raise ValueError('Lexical matching needs nonempty text for every constraint unit')
    mapping = {}
    for node in validate_graph(payload)['nodes']:
        if not is_query(node):
            continue
        if not isinstance(node.get('text'), str) or not node['text'].strip():
            raise ValueError('Lexical matching needs query text; otherwise provide query_unit_ids')
        mapping[node['id']] = [unit['unit_id'] for unit in units if unit_matches(unit['text'], node['text'], rule)]
    return {**annotation, 'query_unit_ids': mapping}
