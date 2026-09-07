import numpy as np
import pandas as pd
from dse_research_utils.ml.permutation import heldout_permutation_deltas
from sklearn.metrics import (
    average_precision_score,
    log_loss,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)


def score_metrics(y_true, p_valid):
    """
    Compute validation metrics: AUC, AP, log loss, ROC curve.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        True binary labels.
    p_valid : array-like of shape (n_samples,)
        Predicted probabilities or scores.

    Returns
    -------
    p_valid_auc : float
        Area Under the ROC Curve.
    p_valid_ap : float
        Average Precision score.
    p_valid_ll : float
        Log loss.
    p_valid_fpr : array-like of shape (n_thresholds,)
        False Positive Rates for ROC curve.
    p_valid_tpr : array-like of shape (n_thresholds,)
        True Positive Rates for ROC curve.
    p_valid_thresholds : array-like of shape (n_thresholds,)
        Thresholds used to compute ROC curve.
    """
    p_valid_auc = roc_auc_score(y_true, p_valid)
    p_valid_ap = average_precision_score(y_true, p_valid)
    p_valid_ll = log_loss(y_true, p_valid, labels=[0, 1])
    p_valid_fpr, p_valid_tpr, p_valid_thresholds = roc_curve(y_true, p_valid)
    return (
        p_valid_auc,
        p_valid_ap,
        p_valid_ll,
        p_valid_fpr,
        p_valid_tpr,
        p_valid_thresholds,
    )


def _as_binary_y(y_true):
    y = np.asarray(y_true)
    if np.isnan(y).any():
        raise ValueError("y_true contains NaNs; binarize/impute first.")
    uniq = np.unique(y)
    if (
        not np.array_equal(uniq, [0, 1])
        and not np.array_equal(uniq, [0])
        and not np.array_equal(uniq, [1])
    ):
        raise ValueError(
            f"y_true must be 0/1 for these metrics. Found labels: {uniq[:20]}"
        )
    return y.astype(np.int8)


def precision_recall_at_k(y_true, p_valid, K: int = 10000):
    """
    Compute precision and recall at top K predictions.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        True binary labels.
    p_valid : array-like of shape (n_samples,)
        Predicted probabilities or scores.
    K : int
        Number of top predictions to consider.

    Returns
    -------
    precision_at_k : float
        Precision at top K predictions.
    recall_at_k : float
        Recall at top K predictions.
    tp : int
        Number of true positives in top K.
    fp : int
        Number of false positives in top K.
    n_pos : int
        Total number of positive samples.
    K : int
        Number of top predictions considered.
    """
    y = _as_binary_y(y_true)
    p = np.asarray(p_valid)

    n = len(y)
    K = min(int(K), n)

    order = np.argsort(-p)
    y_top = y[order[:K]]

    tp = int(y_top.sum())
    fp = int(K - tp)
    n_pos = int((y == 1).sum())

    precision = tp / K if K else 0.0
    recall = tp / n_pos if n_pos else 0.0
    return precision, recall, tp, fp, n_pos, K


def precision_recall_at_threshold(y_true, p_valid, thr: float = 0.01):
    """
    Compute precision and recall at a given threshold.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        True binary labels.
    p_valid : array-like of shape (n_samples,)
        Predicted probabilities or scores.
    thr : float
        Threshold for converting predicted probabilities to binary predictions.

    Returns
    -------
    prec : float
        Precision at the given threshold.
    rec : float
        Recall at the given threshold.
    f1 : float
        F1-score at the given threshold.
    """
    y = _as_binary_y(y_true)
    p = np.asarray(p_valid)

    y_hat = (p >= thr).astype(np.int8)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y, y_hat, average="binary", pos_label=1, zero_division=0
    )
    return float(prec), float(rec), float(f1)


def get_metrics(y_true, p_valid, K: int = 10000, thr: float = 0.01):
    """
    Build a DataFrame of validation metrics.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        True binary labels.
    p_valid : array-like of shape (n_samples,)
        Predicted probabilities or scores.
    K : int
        Number of top predictions to consider for precision/recall at K.
    thr : float
        Threshold for precision/recall calculation.
    tp : int
        Number of true positives in top K.
    fp : int
        Number of false positives in top K.
    n_pos : int
        Total number of positive samples.
    Returns
    -------
    metrics_df : pd.DataFrame
        DataFrame containing validation metrics.
    """

    (
        p_valid_auc,
        p_valid_ap,
        p_valid_ll,
        p_valid_fpr,
        p_valid_tpr,
        p_valid_thresholds,
    ) = score_metrics(y_true, p_valid)

    precision_at_k, recall_at_k, tp, fp, n_pos, K = precision_recall_at_k(
        y_true, p_valid, K=K
    )

    prec, rec, f1 = precision_recall_at_threshold(y_true, p_valid, thr=thr)

    df = pd.DataFrame(
        {
            "metric": [
                "Validation AUC",
                "Validation AP",
                "Validation log loss",
                f"Precision at {K}",
                f"Recall at {K}",
                f"Precision (threshold={thr})",
                f"Recall (threshold={thr})",
            ],
            "value": [
                p_valid_auc,
                p_valid_ap,
                p_valid_ll,
                precision_at_k,
                recall_at_k,
                prec,
                rec,
            ],
        }
    )

    return df, p_valid_fpr, p_valid_tpr, p_valid_thresholds, tp, fp, n_pos


def build_explain_set(
    booster,
    X_valid,
    y_valid,
    categorical,
    n_neg_rand=100_000,
    n_neg_hard=100_000,
    seed=42,
):
    """
    Build a validation set for explanation by combining all positives,
    a random sample of negatives, and a sample of hard negatives (highest predicted
    probabilities among negatives).

    Parameters
    ----------
    booster : lightgbm.Booster
        Trained LightGBM booster.
    X_valid : pd.DataFrame
        Validation feature set.
    y_valid : pd.Series
        Validation target values.
    categorical : list of str
        List of categorical feature names.
    n_neg_rand : int
        Number of random negatives to include.
    n_neg_hard : int
        Number of hard negatives to include.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    X_explain : pd.DataFrame
        Explanation feature set.
    y_explain : pd.Series
        Explanation target values.
    """
    rng = np.random.default_rng(seed)

    yv = np.asarray(y_valid)
    idx_pos = np.flatnonzero(yv == 1)
    idx_neg = np.flatnonzero(yv == 0)

    # predict once on valid to pick hard negatives
    p_valid = booster.predict(X_valid, num_iteration=booster.best_iteration)

    # random negatives
    n_neg_rand = min(n_neg_rand, idx_neg.size)
    idx_neg_rand = rng.choice(idx_neg, size=n_neg_rand, replace=False)

    # hard negatives (top predicted p among negatives)
    n_neg_hard = min(n_neg_hard, idx_neg.size)
    if n_neg_hard > 0:
        p_neg = p_valid[idx_neg]
        hard_local = np.argpartition(p_neg, -n_neg_hard)[-n_neg_hard:]
        idx_neg_hard = idx_neg[hard_local]
    else:
        # argpartition with kth=0 on an empty array would fail, and
        # the [-0:] slice would return everything — neither is correct
        # when the user asked for zero hard negatives.
        idx_neg_hard = np.empty(0, dtype=idx_neg.dtype)

    idx = np.unique(np.concatenate([idx_pos, idx_neg_rand, idx_neg_hard]))
    rng.shuffle(idx)

    X_eval = X_valid.iloc[idx].astype(np.float64).replace({pd.NA: np.nan}).copy()
    X_eval[categorical] = X_eval[categorical].astype("category")
    y_eval = pd.Series(yv[idx], index=X_valid.index[idx])
    return X_eval, y_eval


def ap_scorer(estimator, X, y):
    """
    Average precision scorer for sklearn's cross-validation and hyperparameter tuning utilities.
    """
    proba = estimator.predict_proba(X)[:, 1]
    return average_precision_score(y, proba)


class LGBMEstimator:
    """
    A wrapper for a LightGBM booster to provide sklearn-like interface.
    This is needed because we train the LightGBM model using its native API,
    but we want to use it with sklearn utilities like permutation importance.
    """

    def __init__(self, booster, threshold=0.5):
        self.booster = booster
        self.threshold = threshold

    def fit(self, X, y=None):
        return self

    def _predict_p1(self, X):
        # Use the early-stopped model size
        return self.booster.predict(X, num_iteration=self.booster.best_iteration)

    # ap_scorer calls predict_proba(),
    def predict_proba(self, X):
        p1 = self._predict_p1(X)
        p0 = 1.0 - p1
        return np.column_stack([p0, p1])

    def predict(self, X):
        p1 = self._predict_p1(X)
        return (p1 >= self.threshold).astype(int)


def group_permutation_importance(
    estimator,
    X: pd.DataFrame,
    y: pd.Series,
    groups: dict[str, list[str]],
    scorer=average_precision_score,
    n_repeats: int = 5,
    random_state: int = 0,
    use_predict_proba: bool = True,
):
    """
    Compute group permutation importance.

    Parameters
    ----------
    estimator : object
        Must implement predict_proba(X) -> (n,2) or predict(X) -> (n,).
        Your LGBMWrapper works.
    X : DataFrame
        Evaluation data. Every row is held out and scored; the explanation
        sample built by ``build_explain_set`` defines that population and
        its prevalence.
    y : Series/array
        Labels (0/1).
    groups : dict
        Mapping group_name -> list of column names to permute together.
    scorer : callable
        For AP, pass average_precision_score.
    n_repeats : int
        Permutation repeats per group.
    random_state : int
        Seed.
    use_predict_proba : bool
        If True, scorer uses predict_proba[:,1], else uses predict.

    Returns
    -------
    DataFrame with mean/std importance per group (higher = more important).
    Importance is measured as decrease in score when permuted: (baseline - permuted).

    Notes
    -----
    The evaluation is delegated to
    ``dse_research_utils.ml.permutation.heldout_permutation_deltas``. This
    wrapper owns every project decision it needs:

    - the donor plans, drawn sequentially group-by-group then repeat-by-repeat
      from ``np.random.default_rng(random_state)``, so a seed reproduces the
      previous permutations exactly;
    - dropping features absent from ``X`` and skipping groups left empty,
      before any random number is drawn for them;
    - the positive-class probability column, the average-precision scorer and
      the ``higher_is_better`` direction that makes a positive importance a
      *decrease* in score;
    - the output schema, group identifiers, descending rank and the
      ``ddof=1`` standard deviation (0.0 for a single repeat).

    The shared evaluator scores in float64 and permutes through the column's
    own array, so categorical and nullable dtypes survive a permutation
    instead of being flattened by ``to_numpy()``.
    """
    rng = np.random.default_rng(random_state)

    column_blocks: dict[str, list[str]] = {}
    donor_indices: dict[str, np.ndarray] = {}
    for gname, cols in groups.items():
        cols = [c for c in cols if c in X.columns]
        if len(cols) == 0:
            continue
        column_blocks[gname] = cols
        # One permutation per repeat, applied to every column in the group.
        plan = np.empty((n_repeats, len(X)), dtype=np.intp)
        for repeat in range(n_repeats):
            plan[repeat] = rng.permutation(len(X))
        donor_indices[gname] = plan

    if not column_blocks:
        return pd.DataFrame(
            columns=[
                "rank",
                "group",
                "n_features",
                "features",
                "baseline_score",
                "importance_mean",
                "importance_std",
            ]
        )

    def _predict(fitted, frame: pd.DataFrame) -> np.ndarray:
        if use_predict_proba:
            return fitted.predict_proba(frame)[:, 1]
        return fitted.predict(frame)

    def _score(target: np.ndarray, prediction: np.ndarray) -> float:
        return float(scorer(target, prediction))

    result = heldout_permutation_deltas(
        estimator,
        X,
        np.asarray(y),
        column_blocks,
        donor_indices=donor_indices,
        predict=_predict,
        score=_score,
        score_direction="higher_is_better",
    )

    results = [
        {
            "group": gname,
            "n_features": len(cols),
            "features": cols,
            "baseline_score": result.baseline_score,
            "importance_mean": float(np.mean(result.deltas[gname])),
            "importance_std": (
                float(np.std(result.deltas[gname], ddof=1)) if n_repeats > 1 else 0.0
            ),
        }
        for gname, cols in column_blocks.items()
    ]

    out = pd.DataFrame(results).sort_values("importance_mean", ascending=False)
    out = out.reset_index(drop=True)
    out.insert(0, "rank", np.arange(1, len(out) + 1))
    return out
