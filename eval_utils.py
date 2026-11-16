"""
Evaluation helpers shared by the RF baseline and the GINE model.

Primary metric : macro-average ROC-AUC over tasks (labels with a single class in the evaluated set are skipped).
Secondary      : macro ROC-AUC over the "organ-system" subset (defined BEFORE seeing any model result).
Uncertainty    : 95% bootstrap CI over test molecules (reflects the small test set, NOT the choice of split).
"""
import numpy as np
from sklearn.metrics import roc_auc_score

# MedDRA classes that describe clinical/administrative categories rather than an organ-system response.
# This subset definition is a project decision made before running any model.
NON_ORGAN_SYSTEM = [
    "Social circumstances",
    "Surgical and medical procedures",
    "Investigations",
    "Injury, poisoning and procedural complications",
    "Pregnancy, puerperium and perinatal conditions",
    "Congenital, familial and genetic disorders",
]


def organ_system_tasks(tasks):
    return [t for t in tasks if t not in NON_ORGAN_SYSTEM]


def per_task_auc(y_true, y_score, tasks):
    """y_true/y_score: [n_molecules, n_tasks]. NaN labels are ignored. Returns {task: auc or nan}."""
    out = {}
    for j, t in enumerate(tasks):
        yt, ys = y_true[:, j], y_score[:, j]
        m = ~np.isnan(yt)
        yt, ys = yt[m], ys[m]
        out[t] = float(roc_auc_score(yt, ys)) if len(np.unique(yt)) == 2 else float("nan")
    return out


def macro_auc(auc_dict, subset=None):
    vals = [v for t, v in auc_dict.items() if (subset is None or t in subset) and not np.isnan(v)]
    return float(np.mean(vals)) if vals else float("nan")


def bootstrap_macro_ci(y_true, y_score, tasks, subset=None, n_boot=500, seed=0):
    """95% percentile bootstrap CI of the macro AUC, resampling test molecules with replacement."""
    rng = np.random.RandomState(seed)
    n = len(y_true)
    vals = []
    for _ in range(n_boot):
        idx = rng.randint(0, n, n)
        vals.append(macro_auc(per_task_auc(y_true[idx], y_score[idx], tasks), subset))
    vals = np.array([v for v in vals if not np.isnan(v)])
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def summarize(y_true, y_score, tasks, with_ci=False):
    """Everything we report for one (model, split, set): per-task AUC and macro AUC (all / organ-system)."""
    aucs = per_task_auc(y_true, y_score, tasks)
    organ = organ_system_tasks(tasks)
    res = {
        "per_task_auc": aucs,
        "macro_auc_all": macro_auc(aucs),
        "macro_auc_organ": macro_auc(aucs, organ),
        "n_tasks_all": int(sum(not np.isnan(v) for v in aucs.values())),
    }
    if with_ci:
        res["macro_auc_all_ci95"] = bootstrap_macro_ci(y_true, y_score, tasks)
        res["macro_auc_organ_ci95"] = bootstrap_macro_ci(y_true, y_score, tasks, organ)
    return res
