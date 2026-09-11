"""Validate graph structure without inferring edges or changing node IDs."""

from collections import defaultdict, deque
import re


EDGE_KINDS = (
    'constraint_use', 'evidence_derived', 'prior_knowledge_derived', 'failure_derived',
)


def graph_of(payload):
    if not isinstance(payload, dict):
        raise ValueError('A graph record must be an object')
    graph = payload.get('graph', payload.get('final_ground_truth_dag', payload))
    if not isinstance(graph, dict) or not isinstance(graph.get('nodes'), list) or not isinstance(graph.get('edges'), list):
        raise ValueError('Expected graph.nodes and graph.edges lists')
    return graph


def is_query(node):
    if node['id'] in {'Q0', 'Answer', 'Prior_knowledge'}:
        return False
    return str(node.get('type', '')).lower() == 'query' or bool(re.fullmatch(r'q\d+', node['id']))


def topological_order(ids, pairs):
    children = defaultdict(set)
    degree = dict.fromkeys(ids, 0)
    for source, target in pairs:
        if source not in degree or target not in degree:
            raise ValueError('An edge references a missing node')
        if target not in children[source]:
            children[source].add(target)
            degree[target] += 1
    queue = deque(sorted(node for node, count in degree.items() if count == 0))
    ordered = []
    while queue:
        node = queue.popleft()
        ordered.append(node)
        for child in sorted(children[node]):
            degree[child] -= 1
            if degree[child] == 0:
                queue.append(child)
    if len(ordered) != len(degree):
        raise ValueError('Input contains a directed cycle')
    return ordered


def validate_graph(payload):
    graph = graph_of(payload)
    nodes = {}
    for node in graph['nodes']:
        if not isinstance(node, dict) or not isinstance(node.get('id'), str) or not node['id']:
            raise ValueError('Every node needs a nonempty string id')
        if node['id'] in nodes:
            raise ValueError('Duplicate node id')
        nodes[node['id']] = node
    for edge in graph['edges']:
        if not isinstance(edge, dict):
            raise ValueError('Edges must be objects')
        if edge.get('source') not in nodes or edge.get('target') not in nodes:
            raise ValueError('An edge references a missing node')
        if edge.get('edge_kind') not in EDGE_KINDS:
            raise ValueError('Each edge needs one of the four supported edge_kind values')
    topological_order(nodes, {(e['source'], e['target']) for e in graph['edges']})
    return graph


def edge_sets(payload):
    """Untyped pairs plus per-kind pairs; first kind wins on a duplicated pair."""
    graph = validate_graph(payload)
    kinds = {kind: set() for kind in EDGE_KINDS}
    owners = {}
    conflicts = set()
    for edge in graph['edges']:
        pair = (edge['source'], edge['target'])
        kind = edge['edge_kind']
        if pair in owners:
            if owners[pair] != kind:
                conflicts.add(pair)
            continue
        owners[pair] = kind
        kinds[kind].add(pair)
    return set(owners), kinds, len(conflicts)
