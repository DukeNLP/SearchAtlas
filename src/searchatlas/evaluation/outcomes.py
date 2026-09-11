"""Within-agent outcome association using question-held-out normalization."""

from collections import defaultdict
import math
from statistics import mean


STAGES = {
    'Topology': [('topology', 1.0)],
    '+ Grounding': [('topology', 1.0), ('grounding', 1.0)],
    '+ PK reliance': [('topology', 1.0), ('grounding', 1.0), ('pk_risk', -1.0)],
    'Grounding only': [('grounding', 1.0)],
    'PK only': [('pk_risk', -1.0)],
    'w/o topology': [('grounding', 1.0), ('pk_risk', -1.0)],
    'w/o grounding': [('topology', 1.0), ('pk_risk', -1.0)],
    'w/o PK': [('topology', 1.0), ('grounding', 1.0)],
}


def _inputs(labels, scores):
    if not labels or len(labels) != len(scores):
        raise ValueError('Labels and scores must be nonempty and have equal lengths')
    if any(label not in (0, 1) for label in labels):
        raise ValueError('Correctness labels must be binary (0 or 1)')
    if any(value is None or not math.isfinite(float(value)) for value in scores):
        raise ValueError('Scores must be finite numbers')


def roc_auc(labels, scores):
    _inputs(labels, scores)
    positive, negative = sum(labels), len(labels) - sum(labels)
    if not positive or not negative:
        return None
    ordered = sorted(range(len(scores)), key=scores.__getitem__)
    rank_sum, i = 0.0, 0
    while i < len(ordered):
        j = i + 1
        while j < len(ordered) and scores[ordered[j]] == scores[ordered[i]]:
            j += 1
        rank_sum += sum(labels[ordered[k]] for k in range(i, j)) * ((i + 1 + j) / 2)
        i = j
    return (rank_sum - positive * (positive + 1) / 2) / (positive * negative)


def average_precision(labels, scores, method='rank'):
    """Rank AP preserves tie order; grouped AP integrates tied score thresholds."""
    _inputs(labels, scores)
    if method not in {'rank', 'grouped'}:
        raise ValueError('AP method must be rank or grouped')
    positive = sum(labels)
    if not positive:
        return None
    ordered = sorted(range(len(scores)), key=scores.__getitem__, reverse=True)
    hits, total, i = 0, 0.0, 0
    while i < len(ordered):
        j = i + 1
        if method == 'grouped':
            while j < len(ordered) and scores[ordered[j]] == scores[ordered[i]]:
                j += 1
        added = sum(labels[ordered[k]] for k in range(i, j))
        hits += added
        total += added * hits / j
        i = j
    return total / positive


def binary_results(labels, predictions):
    _inputs(labels, predictions)
    if any(value not in (0, 1) for value in predictions):
        raise ValueError('Predictions must be binary (0 or 1)')
    tp = sum(y == p == 1 for y, p in zip(labels, predictions))
    tn = sum(y == p == 0 for y, p in zip(labels, predictions))
    fp = sum(y == 0 and p == 1 for y, p in zip(labels, predictions))
    fn = sum(y == 1 and p == 0 for y, p in zip(labels, predictions))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    f1 = 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else 0.0
    negative_f1 = 2 * tn / (2 * tn + fp + fn) if tn + fp + fn else 0.0
    return dict(n=len(labels), tp=tp, tn=tn, fp=fp, fn=fn,
                accuracy=(tp + tn) / len(labels), precision=precision, recall=recall,
                f1=f1, macro_f1=(f1 + negative_f1) / 2,
                balanced_accuracy=(recall + specificity) / 2 if len(set(labels)) == 2 else None,
                auc=roc_auc(labels, predictions), ap=average_precision(labels, predictions, 'grouped'))


def question_folds(rows, n_folds=5):
    if n_folds < 2:
        raise ValueError('At least two folds are required')
    questions = sorted({row['question_id'] for row in rows})
    if len(questions) < 2:
        raise ValueError('Held-out scoring needs at least two distinct questions per agent/regime')
    return {question: i % n_folds for i, question in enumerate(questions)}


def normalized_scores(train, rows, terms):
    """Fit each metric's mean/population SD on training rows only."""
    if not train:
        raise ValueError('Training rows must not be empty')
    scores = [0.0] * len(rows)
    for name, sign in terms:
        values = [float(row[name]) for row in train]
        center = mean(values)
        sd = math.sqrt(mean((value - center) ** 2 for value in values)) or 1.0
        for i, row in enumerate(rows):
            scores[i] += sign * (float(row[name]) - center) / sd
    return scores


def heldout_scores(rows, terms, n_folds=5):
    folds = question_folds(rows, n_folds)
    result = [0.0] * len(rows)
    for fold in sorted(set(folds.values())):
        train = [row for row in rows if folds[row['question_id']] != fold]
        indices = [i for i, row in enumerate(rows) if folds[row['question_id']] == fold]
        scores = normalized_scores(train, [rows[i] for i in indices], terms)
        for i, score in zip(indices, scores):
            result[i] = score
    return result


def best_threshold(labels, scores):
    _inputs(labels, scores)
    if len(set(labels)) != 2:
        return None
    values = sorted(set(scores))
    candidates = [values[0]] + [a / 2 + b / 2 for a, b in zip(values, values[1:])]
    candidates.append(math.nextafter(values[-1], math.inf))
    best, accuracy = candidates[0], -1.0
    for threshold in candidates:
        prediction = [int(score >= threshold) for score in scores]
        score = binary_results(labels, prediction)['balanced_accuracy']
        if score > accuracy:
            best, accuracy = threshold, score
    return best


def heldout_predictions(rows, n_folds=5):
    folds = question_folds(rows, n_folds)
    result = []
    terms = STAGES['+ PK reliance']
    for fold in sorted(set(folds.values())):
        train = [row for row in rows if folds[row['question_id']] != fold]
        test = [row for row in rows if folds[row['question_id']] == fold]
        threshold = best_threshold([row['correct'] for row in train], normalized_scores(train, train, terms))
        for row, score in zip(test, normalized_scores(train, test, terms)):
            result.append({'case_id': row['case_id'], 'correct': row['correct'],
                           'fold': fold, 'score': score, 'threshold': threshold,
                           'prediction': None if threshold is None else int(score >= threshold)})
    return result


def _mean_or_none(values):
    finite = [value for value in values if value is not None]
    return mean(finite) if finite else None


def outcome_results(rows, n_folds=5, ap_method='rank'):
    if not rows or len({row['case_id'] for row in rows}) != len(rows):
        raise ValueError('Outcome rows must have unique case IDs and be nonempty')
    cells = defaultdict(list)
    for row in rows:
        for field in ('question_id', 'dataset', 'regime', 'model'):
            if not isinstance(row.get(field), str) or not row[field]:
                raise ValueError(f'Each outcome row needs {field}')
        _inputs([row['correct']] * 3, [row.get(name) for name in ('topology', 'grounding', 'pk_risk')])
        cells[(row['dataset'], row['regime'], row['model'])].append(row)
    detail, threshold_results = [], []
    for (dataset, regime, model), cell in cells.items():
        meta = dict(dataset=dataset, regime=regime, model=model)
        labels = [row['correct'] for row in cell]
        for stage, terms in STAGES.items():
            scores = heldout_scores(cell, terms, n_folds)
            detail.append({**meta, 'stage': stage, 'n': len(cell), 'accuracy': mean(labels),
                           'auc': roc_auc(labels, scores), 'ap': average_precision(labels, scores, ap_method)})
        predictions = heldout_predictions(cell, n_folds)
        valid = [row for row in predictions if row['prediction'] is not None]
        threshold_results.append({**meta, 'n_total': len(cell), 'n_evaluated': len(valid),
                                  'metrics': binary_results([r['correct'] for r in valid], [r['prediction'] for r in valid]) if valid else None,
                                  'predictions': predictions})
    groups = defaultdict(list)
    for row in detail:
        groups[(row['dataset'], row['regime'], row['stage'])].append(row)
    macro = []
    for (dataset, regime, stage), group in groups.items():
        macro.append(dict(dataset=dataset, regime=regime, stage=stage, n=sum(row['n'] for row in group),
                          models=len(group), auc_models=sum(row['auc'] is not None for row in group),
                          ap_models=sum(row['ap'] is not None for row in group),
                          auc=_mean_or_none(row['auc'] for row in group), ap=_mean_or_none(row['ap'] for row in group)))
    return {'per_agent': detail, 'macro': macro, 'thresholds': threshold_results,
            'protocol': {'folds': n_folds, 'fold_key': 'question_id', 'normalization': 'training_population_sd',
                         'ap_method': ap_method, 'grouping': ['dataset', 'regime', 'model']}}
