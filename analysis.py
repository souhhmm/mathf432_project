import argparse
import csv
import hashlib
import itertools
import json
from pathlib import Path

import matplotlib
import numpy as np
import scipy
from scipy import linalg, optimize, special, stats

from plot_results import draw_figures

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "uci697.csv"
OUTCOMES = ("Dropout", "Enrolled", "Graduate")
SEED = 432
AGE = "Age at enrollment"
ADMISSION = "Admission grade"
PREVIOUS = "Previous qualification (grade)"
SCHOLARSHIP = "Scholarship holder"
TUITION = "Tuition fees up to date"


def semester(term, field):
    label = ("1st", "2nd")[term - 1]
    return f"Curricular units {label} sem ({field})"


def load_data():
    with DATA.open(encoding="utf-8-sig", newline="") as handle:
        header, *rows = csv.reader(handle)
    if (
        len(rows) != 4424
        or len(header) != 37
        or len(set(header)) != 37
        or header[-1] != "Target"
        or any(len(row) != len(header) for row in rows)
    ):
        raise ValueError("Unexpected shape or header for the UCI 697 release")

    outcomes = np.array([row[-1] for row in rows])
    matrix = np.array([row[:-1] for row in rows], dtype=float)
    if not np.all(np.isfinite(matrix)) or set(outcomes) != set(OUTCOMES):
        raise ValueError("Unexpected missing values or outcome labels")

    data = dict(zip(header[:-1], matrix.T))
    audit = {
        "records": len(rows),
        "features": len(data),
        "sha256": hashlib.sha256(DATA.read_bytes()).hexdigest(),
        "missing_cells": sum(field == "" for row in rows for field in row),
        "duplicate_rows": len(rows) - len({tuple(row) for row in rows}),
        "programmes": len(np.unique(data["Course"])),
        "columns": {
            name: {
                "min": min(values),
                "max": max(values),
                "unique": len(np.unique(values)),
            }
            for name, values in data.items()
        },
    }
    for term in (1, 2):
        grades = data[semester(term, "grade")]
        passed = data[semester(term, "approved")]
        audit[f"semester_{term}"] = {
            "zero_grade": int(np.sum(grades == 0)),
            "no_passed_units": int(np.sum(passed == 0)),
            "zero_grade_pass_mismatches": int(np.sum((grades == 0) != (passed == 0))),
            "zero_units": int(np.sum(data[semester(term, "enrolled")] == 0)),
            "zero_evaluations": int(np.sum(data[semester(term, "evaluations")] == 0)),
        }
    return data, outcomes, audit


def wilson(successes, total):
    result = stats.binomtest(successes, total)
    interval = result.proportion_ci(method="wilson")
    return [interval.low, interval.high]


def summarize(values):
    n = len(values)
    mean = np.mean(values)
    sd = np.std(values, ddof=1)
    lo, hi = stats.t.interval(0.95, n - 1, loc=mean, scale=sd / np.sqrt(n))
    return {
        "n": n,
        "mean": mean,
        "sd": sd,
        "mean_ci": [lo, hi],
        "quantiles": np.quantile(values, [0, 0.25, 0.50, 0.75, 1]).tolist(),
    }


def test_result(result):
    return {"statistic": float(result.statistic), "p": float(result.pvalue)}


def describe(data, outcomes):
    profiles = {}
    grade1 = data[semester(1, "grade")]
    grade2 = data[semester(2, "grade")]
    for label in OUTCOMES:
        group = outcomes == label
        count = int(np.sum(group))
        profiles[label] = {
            "n": count,
            "share": count / len(outcomes),
            "share_ci": wilson(count, len(outcomes)),
            "age": summarize(data[AGE][group]),
            "admission": summarize(data[ADMISSION][group]),
            "positive_grade1": summarize(grade1[group & (grade1 > 0)]),
            "grade1_zero": int(np.sum(group & (grade1 == 0))),
            "grade2_zero": int(np.sum(group & (grade2 == 0))),
            "no_evaluations1": int(
                np.sum(group & (data[semester(1, "evaluations")] == 0))
            ),
        }
    return profiles


def compare_grades(data, outcomes):
    groups = [data[ADMISSION][outcomes == label] for label in OUTCOMES]
    results = {
        "shapiro": {
            label: test_result(stats.shapiro(group))
            for label, group in zip(OUTCOMES, groups)
        },
        "brown_forsythe": test_result(stats.levene(*groups, center="median")),
        "welch_anova": test_result(stats.f_oneway(*groups, equal_var=False)),
        "kruskal_wallis": test_result(stats.kruskal(*groups)),
    }
    group_sizes = np.array([len(group) for group in groups])
    group_variances = np.array([np.var(group, ddof=1) for group in groups])
    weights = group_sizes / group_variances
    group_count = len(groups)
    results["welch_anova"]["df"] = [
        group_count - 1,
        (group_count**2 - 1)
        / (3 * np.sum((1 - weights / weights.sum()) ** 2 / (group_sizes - 1))),
    ]

    contrasts = []
    for i, j in itertools.combinations(range(group_count), 2):
        left, right = groups[i], groups[j]
        test = stats.ttest_ind(left, right, equal_var=False)
        degrees_of_freedom = len(left) + len(right) - 2
        pooled_sd = np.sqrt(
            (
                (len(left) - 1) * group_variances[i]
                + (len(right) - 1) * group_variances[j]
            )
            / degrees_of_freedom
        )
        delta = np.mean(left) - np.mean(right)
        correction = 1 - 3 / (4 * (len(left) + len(right)) - 9)
        contrasts.append(
            {
                "groups": [OUTCOMES[i], OUTCOMES[j]],
                "mean_difference": delta,
                "ci": list(test.confidence_interval()),
                "welch_t": test_result(test),
                "df": test.df,
                "hedges_g": delta / pooled_sd * correction,
            }
        )
    ordered = sorted(contrasts, key=lambda row: row["welch_t"]["p"])
    adjusted = 0.0
    for rank, row in enumerate(ordered):
        adjusted = max(adjusted, (len(ordered) - rank) * row["welch_t"]["p"])
        row["holm_p"] = min(1.0, adjusted)
    results["contrasts"] = contrasts
    return results


def compare_admission_profiles(data, outcomes):
    """Compare the two outcome groups on a joint admission profile."""
    features = (ADMISSION, PREVIOUS, AGE)
    matrix = np.column_stack([data[feature] for feature in features])
    dropout = matrix[outcomes == "Dropout"]
    graduate = matrix[outcomes == "Graduate"]
    n_d, n_g = len(dropout), len(graduate)
    p = len(features)
    mean_d, mean_g = dropout.mean(axis=0), graduate.mean(axis=0)
    cov_d = np.cov(dropout, rowvar=False)
    cov_g = np.cov(graduate, rowvar=False)
    difference = mean_d - mean_g

    n = n_d + n_g
    pooled = ((n_d - 1) * cov_d + (n_g - 1) * cov_g) / (n - 2)
    distance_squared = difference @ linalg.solve(pooled, difference, assume_a="pos")
    t_squared = (n_d * n_g / n) * distance_squared
    df = n - p - 1
    f = df * t_squared / (p * (n - 2))

    # a large-sample wald check lets each group have its own covariance.
    mean_covariance = cov_d / n_d + cov_g / n_g
    wald = difference @ linalg.solve(mean_covariance, difference, assume_a="pos")
    return {
        "features": list(features),
        "groups": {
            "Dropout": {
                "n": n_d,
                "mean": mean_d.tolist(),
                "sd": np.sqrt(cov_d.diagonal()).tolist(),
                "covariance": cov_d.tolist(),
            },
            "Graduate": {
                "n": n_g,
                "mean": mean_g.tolist(),
                "sd": np.sqrt(cov_g.diagonal()).tolist(),
                "covariance": cov_g.tolist(),
            },
        },
        "mean_difference": difference.tolist(),
        "mahalanobis_d_squared": distance_squared,
        "mahalanobis_d": np.sqrt(distance_squared),
        "hotelling_t_squared": t_squared,
        "pooled_f_test": {"statistic": f, "df": [p, df], "p": stats.f.sf(f, p, df)},
        "unequal_covariance_wald": {
            "statistic": wald,
            "df": p,
            "p": stats.chi2.sf(wald, p),
        },
    }


def compare_finances(data, outcomes):
    # row order is 0 then 1. the target column order is outcomes.
    tuition = data[TUITION]
    counts = np.array(
        [
            [np.sum((tuition == status) & (outcomes == label)) for label in OUTCOMES]
            for status in (0, 1)
        ]
    )
    test = stats.chi2_contingency(counts)
    tuition_results = {
        "observed": counts.tolist(),
        "expected": test.expected_freq.tolist(),
        "chi_square": test_result(test),
        "df": test.dof,
        "cramers_v": np.sqrt(test.statistic / counts.sum()),
    }

    # each scholarship group includes all three outcomes in its denominator.
    scholarship = data[SCHOLARSHIP]
    group_sizes = [int(np.sum(scholarship == value)) for value in (1, 0)]
    dropout_counts = [
        int(np.sum((scholarship == value) & (outcomes == "Dropout")))
        for value in (1, 0)
    ]
    rate_with, rate_without = (
        count / size for count, size in zip(dropout_counts, group_sizes)
    )
    ci_with, ci_without = (
        wilson(count, size) for count, size in zip(dropout_counts, group_sizes)
    )
    pooled_rate = sum(dropout_counts) / sum(group_sizes)
    standard_error = np.sqrt(
        pooled_rate * (1 - pooled_rate) * (1 / group_sizes[0] + 1 / group_sizes[1])
    )
    difference = rate_with - rate_without
    z = difference / standard_error
    # newcombe's unpaired difference interval combines two wilson intervals.
    difference_ci = [
        difference - np.hypot(rate_with - ci_with[0], ci_without[1] - rate_without),
        difference + np.hypot(ci_with[1] - rate_with, rate_without - ci_without[0]),
    ]
    return {
        "tuition": tuition_results,
        "scholarship": {
            "dropout_counts": dropout_counts,
            "group_sizes": group_sizes,
            "dropout_shares": [rate_with, rate_without],
            "difference": difference,
            "difference_ci": difference_ci,
            "z": z,
            "p": 2 * stats.norm.sf(abs(z)),
        },
    }


def rotated_loadings(loadings):
    """Apply an orthogonal varimax rotation."""
    p, k = loadings.shape
    rotation = np.eye(k)
    previous = 0.0
    for _ in range(500):
        rotated = loadings @ rotation
        gradient = rotated**3 - rotated * np.sum(rotated**2, axis=0) / p
        left, singular_values, right = np.linalg.svd(loadings.T @ gradient)
        rotation = left @ right
        objective = singular_values.sum()
        if abs(objective - previous) < 1e-9:
            break
        previous = objective
    rotated = loadings @ rotation
    order = np.argsort(np.sum(rotated**2, axis=0))[::-1]
    rotated = rotated[:, order]

    # give each factor a reproducible sign by making its strongest loading positive.
    strongest = np.argmax(np.abs(rotated), axis=0)
    return rotated * np.sign(rotated[strongest, np.arange(k)])


def explore_factors(data, subset):
    fields = ("enrolled", "evaluations", "approved", "grade")
    labels = [f"Sem. {term} {name}" for term in (1, 2) for name in fields]
    values = np.column_stack(
        [data[semester(term, field)][subset] for term in (1, 2) for field in fields]
    )
    n, p = values.shape
    corr = np.corrcoef(values, rowvar=False)
    inverse = np.linalg.inv(corr)
    partial = -inverse / np.sqrt(np.outer(inverse.diagonal(), inverse.diagonal()))
    np.fill_diagonal(partial, 0)
    corr_squared = corr**2 - np.eye(p)
    per_item_kmo = corr_squared.sum(axis=0) / (corr_squared + partial**2).sum(axis=0)
    kmo = corr_squared.sum() / (corr_squared + partial**2).sum()
    bartlett = -(n - 1 - (2 * p + 5) / 6) * np.linalg.slogdet(corr)[1]
    df = p * (p - 1) // 2

    generator = np.random.default_rng(SEED + 1)
    simulated = np.array(
        [
            np.linalg.eigvalsh(
                np.corrcoef(generator.standard_normal((n, p)), rowvar=False)
            )[::-1]
            for _ in range(499)
        ]
    )
    observed = np.linalg.eigvalsh(corr)[::-1]
    null_cutoff = np.quantile(simulated, 0.95, axis=0)
    retained = next(
        (i for i, (a, b) in enumerate(zip(observed, null_cutoff)) if a <= b), p
    )
    result = {
        "n": n,
        "labels": labels,
        "kmo": kmo,
        "item_kmo": per_item_kmo.tolist(),
        "bartlett": {"statistic": bartlett, "df": df, "p": stats.chi2.sf(bartlett, df)},
        "observed_eigenvalues": observed.tolist(),
        "null_eigenvalues_p95": null_cutoff.tolist(),
        "retained": retained,
        "correlations": corr.tolist(),
    }
    model_df = p * (p + 1) / 2 - (p * (retained + 1) - retained * (retained - 1) / 2)
    if kmo < 0.60 or min(per_item_kmo) < 0.50 or not retained or model_df <= 0:
        return result

    offdiag = np.tril_indices(p, -1)

    def extract(uniqueness):
        eigenvalues, eigenvectors = np.linalg.eigh(corr - np.diag(uniqueness))
        return eigenvectors[:, -retained:] * np.sqrt(
            np.maximum(eigenvalues[-retained:], 0)
        )

    def residuals(uniqueness):
        loadings = extract(uniqueness)
        return (corr - loadings @ loadings.T)[offdiag]

    fit = optimize.least_squares(
        residuals,
        np.clip(1 / inverse.diagonal(), 0.005, 1),
        bounds=(0.005, 1),
        gtol=1e-10,
    )
    if not fit.success:
        raise RuntimeError(f"Minimum-residual factor extraction failed: {fit.message}")
    loadings = rotated_loadings(extract(fit.x))
    result.update(
        {
            "loadings": loadings.tolist(),
            "communalities": np.sum(loadings**2, axis=1).tolist(),
            "variance_share": (np.sum(loadings**2, axis=0) / p).tolist(),
            "offdiag_rms_residual": np.sqrt(np.mean(fit.fun**2)),
            "uniqueness_at_lower_bound": int(np.sum(np.isclose(fit.x, 0.005))),
        }
    )
    return result


def coefficients(
    names, beta, covariance, distribution, transform=lambda values: values
):
    se = np.sqrt(np.diag(covariance))
    critical = distribution.ppf(0.975)
    return {
        name: {
            "estimate": transform(b),
            "ci": [transform(b - critical * s), transform(b + critical * s)],
            "p": 2 * distribution.sf(abs(b / s)),
        }
        for name, b, s in zip(names, beta, se)
    }


def design_matrix(columns, subset, course_levels=None):
    names = ["Intercept", *columns]
    predictors = [values[subset] for values in columns.values()]
    matrix = np.column_stack([np.ones(np.sum(subset)), *predictors])
    if course_levels is not None:
        course, levels = course_levels
        names.extend(f"Programme {int(level)}" for level in levels[1:])
        indicators = course[subset, None] == levels[None, 1:]
        matrix = np.column_stack([matrix, indicators])
    return matrix, names


def regress_grades(data, subset, columns):
    x, names = design_matrix(columns, subset)
    y = data[semester(2, "grade")][subset]
    n, p = x.shape
    beta = np.linalg.lstsq(x, y, rcond=None)[0]
    fitted = x @ beta
    residual = y - fitted
    xtx_inv = np.linalg.inv(x.T @ x)
    leverage = np.sum((x @ xtx_inv) * x, axis=1)
    corrected = residual / (1 - leverage)
    hc3 = xtx_inv @ ((x.T * corrected**2) @ x) @ xtx_inv
    mse = residual @ residual / (n - p)
    r2 = 1 - (residual @ residual) / np.sum((y - y.mean()) ** 2)

    # the studentized breusch-pagan statistic uses an ols auxiliary regression.
    squared = residual**2
    aux_resid = squared - x @ np.linalg.lstsq(x, squared, rcond=None)[0]
    lm = n * (1 - (aux_resid @ aux_resid) / np.sum((squared - squared.mean()) ** 2))
    cook = corrected**2 * leverage / (p * mse)
    correlations = np.atleast_2d(np.corrcoef(x[:, 1:], rowvar=False))
    vif = np.diag(np.linalg.inv(correlations))
    result = {
        "n": n,
        "r_squared": r2,
        "adjusted_r_squared": 1 - (1 - r2) * (n - 1) / (n - p),
        "residual_sd": np.sqrt(mse),
        "coefficients": coefficients(names, beta, hc3, stats.t(n - p)),
        "vif": dict(zip(names[1:], vif)),
        "shapiro_residuals": test_result(stats.shapiro(residual)),
        "breusch_pagan": {"statistic": lm, "df": p - 1, "p": stats.chi2.sf(lm, p - 1)},
        "max_cook_distance": max(cook),
        "n_cook_above_4_over_n": int(np.sum(cook > 4 / n)),
    }
    return result, (fitted, residual)


def fit_logistic(x, y):
    def negative_log_likelihood(beta):
        linear = x @ beta
        return np.sum(np.logaddexp(0, linear) - y * linear)

    def gradient(beta):
        return x.T @ (special.expit(x @ beta) - y)

    def hessian(beta):
        probability = special.expit(x @ beta)
        return (x.T * (probability * (1 - probability))) @ x

    fit = optimize.minimize(
        negative_log_likelihood,
        np.zeros(x.shape[1]),
        method="trust-exact",
        jac=gradient,
        hess=hessian,
        options={"gtol": 1e-7},
    )
    if np.max(np.abs(gradient(fit.x))) > 1e-5:
        raise RuntimeError(f"Logistic regression did not converge: {fit.message}")
    return fit.x, np.linalg.inv(hessian(fit.x))


def fit_lda(x, y):
    """Fit the log-odds coefficients and measure the class covariance difference."""
    negative, positive = x[~y], x[y]
    n0, n1 = len(negative), len(positive)
    mean0, mean1 = negative.mean(axis=0), positive.mean(axis=0)
    cov0, cov1 = np.cov(negative, rowvar=False), np.cov(positive, rowvar=False)
    pooled = ((n0 - 1) * cov0 + (n1 - 1) * cov1) / (n0 + n1 - 2)
    slopes = linalg.solve(pooled, mean1 - mean0, assume_a="pos")
    intercept = -0.5 * ((mean1 + mean0) @ slopes) + np.log(n1 / n0)
    scaled_difference = (cov1 - cov0) / np.sqrt(
        np.outer(pooled.diagonal(), pooled.diagonal())
    )
    return np.r_[intercept, slopes], float(np.sqrt(np.mean(scaled_difference**2)))


def roc_and_precision(y, score):
    order = np.argsort(-score, kind="stable")
    unique_ends = np.r_[np.flatnonzero(np.diff(score[order])), len(y) - 1]
    tp = np.cumsum(y[order])[unique_ends]
    fp = unique_ends + 1 - tp
    tpr = np.r_[0, tp / np.sum(y)]
    fpr = np.r_[0, fp / np.sum(~y)]
    ap = np.diff(tpr) @ (tp / (tp + fp))
    return fpr, tpr, ap


def measure(y, score):
    fpr, tpr, ap = roc_and_precision(y, score)
    positive = score >= 0  # log odds of zero correspond to a 0.5 probability.
    tn = int(np.sum(~y & ~positive))
    fp = int(np.sum(~y & positive))
    fn = int(np.sum(y & ~positive))
    tp = int(np.sum(y & positive))
    return {
        "auc": np.trapezoid(tpr, x=fpr),
        "average_precision": ap,
        "accuracy": (tp + tn) / len(y),
        "balanced_accuracy": 0.5 * (tp / (tp + fn) + tn / (tn + fp)),
        "recall": tp / (tp + fn),
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "confusion": [[tn, fp], [fn, tp]],
    }, (fpr, tpr)


def bootstrap_auc(y, scores, draws=2000):
    generator = np.random.default_rng(SEED + 2)
    positive_rows = np.flatnonzero(y)
    negative_rows = np.flatnonzero(~y)
    positive_count = len(positive_rows)
    negative_count = len(negative_rows)
    baseline_rank_sum = positive_count * (positive_count + 1) / 2
    pair_count = positive_count * negative_count
    aucs = np.empty((draws, len(scores)))
    for draw in range(draws):
        # resample each class separately and use the same rows for every model.
        sampled_positive = generator.choice(positive_rows, positive_count)
        sampled_negative = generator.choice(negative_rows, negative_count)
        sampled = np.r_[sampled_positive, sampled_negative]
        ranks = stats.rankdata(scores[:, sampled], axis=1)
        positive_rank_sum = ranks[:, :positive_count].sum(axis=1)
        aucs[draw] = (positive_rank_sum - baseline_rank_sum) / pair_count
    return aucs


def classify(data, outcomes, train, test, intervals):
    age = data[AGE] / 5
    base = {
        "Age (5 years)": age,
        "Admission grade (10 points)": data[ADMISSION] / 10,
        "Previous grade (10 points)": data[PREVIOUS] / 10,
        "Application order": data["Application order"],
        "Displaced": data["Displaced"],
    }
    passed = data[semester(1, "approved")]
    any_passed = passed > 0
    first_semester = {
        "First-semester units enrolled": data[semester(1, "enrolled")],
        "First-semester units passed": passed,
        "Any first-semester unit passed": any_passed,
        "First-semester grade among passers": (data[semester(1, "grade")] - 13)
        * any_passed,
    }
    course = data["Course"]
    course_levels = np.unique(course[train])
    if not np.all(np.isin(course[test], course_levels)):
        raise ValueError("The test split has a programme missing from training")
    y_train, y_test = outcomes[train] == "Dropout", outcomes[test] == "Dropout"
    result = {
        "train_n": int(train.sum()),
        "test_n": int(test.sum()),
        "train_dropout_n": int(y_train.sum()),
        "test_dropout_n": int(y_test.sum()),
        "programme_reference": int(course_levels[0]),
        "models": {},
    }
    curves = {}
    model_scores = {}
    models = result["models"]
    for name, features in (("A", base), ("B", base | first_semester)):
        x_train, names = design_matrix(features, train, (course, course_levels))
        x_test, _ = design_matrix(features, test, (course, course_levels))
        mean = x_train[:, 1:].mean(axis=0)
        sd = x_train[:, 1:].std(axis=0, ddof=1)
        scale = np.r_[1.0, sd]
        scaled = np.column_stack([x_train[:, 0], (x_train[:, 1:] - mean) / sd])
        if np.linalg.matrix_rank(scaled) != scaled.shape[1]:
            raise ValueError(f"Model {name} has a rank-deficient training design")

        logistic, covariance = fit_logistic(scaled, y_train)
        lda, mismatch = fit_lda(scaled[:, 1:], y_train)
        for method, parameters in (("Logistic", logistic), ("LDA", lda)):
            key = f"{method} {name}"
            beta = parameters / scale
            beta[0] -= mean @ beta[1:]
            score = x_test @ beta
            row, curves[key] = measure(y_test, score)
            model_scores[key] = score
            models[key] = row
            if method == "Logistic":
                # move coefficient covariance back to the original predictor units.
                coefficient_transform = np.diag(1 / scale)
                coefficient_transform[0, 1:] = -mean / sd
                odds_covariance = (
                    coefficient_transform @ covariance @ coefficient_transform.T
                )
                row["odds_ratios"] = coefficients(
                    names, beta, odds_covariance, stats.norm, transform=np.exp
                )
            else:
                row["covariance_rms_difference"] = mismatch

    if intervals:
        aucs = bootstrap_auc(y_test, np.array(list(model_scores.values())))
        aucs_by_model = dict(zip(model_scores, aucs.T))
        for key, draws in aucs_by_model.items():
            models[key]["auc_ci"] = np.quantile(draws, [0.025, 0.975]).tolist()
        gain = aucs_by_model["Logistic B"] - aucs_by_model["Logistic A"]
        result["logistic_auc_gain_ci"] = np.quantile(gain, [0.025, 0.975]).tolist()
    result["logistic_auc_gain"] = (
        models["Logistic B"]["auc"] - models["Logistic A"]["auc"]
    )
    return result, curves


def main(*, results_only=False):
    data, outcomes, audit = load_data()
    profiles = describe(data, outcomes)
    grade_tests = compare_grades(data, outcomes)
    admission_profiles = compare_admission_profiles(data, outcomes)
    finance = compare_finances(data, outcomes)
    graded_both = (data[semester(1, "grade")] > 0) & (data[semester(2, "grade")] > 0)
    factors = explore_factors(data, graded_both)
    baseline_columns = {"First-semester grade": data[semester(1, "grade")]}
    adjusted_columns = {
        **baseline_columns,
        "First-semester units passed": data[semester(1, "approved")],
        "First-semester units enrolled": data[semester(1, "enrolled")],
        "Admission grade (10 points)": data[ADMISSION] / 10,
        "Previous grade (10 points)": data[PREVIOUS] / 10,
        "Age (5 years)": data[AGE] / 5,
    }
    simple, _ = regress_grades(data, graded_both, baseline_columns)
    regression, residuals = regress_grades(data, graded_both, adjusted_columns)

    generator = np.random.default_rng(SEED)
    train = np.zeros(len(outcomes), dtype=bool)
    test = np.zeros_like(train)
    for label in ("Dropout", "Graduate"):
        rows = generator.permutation(np.flatnonzero(outcomes == label))
        n_test = round(len(rows) * 0.25)
        test[rows[:n_test]] = True
        train[rows[n_test:]] = True
    prediction, curves = classify(data, outcomes, train, test, intervals=True)
    evaluated = data[semester(1, "evaluations")] > 0
    sensitivity, _ = classify(
        data, outcomes, train & evaluated, test & evaluated, intervals=True
    )

    result = {
        "audit": audit,
        "profiles": profiles,
        "grade_tests": grade_tests,
        "admission_profiles": admission_profiles,
        "financial_tests": finance,
        "factor_analysis": factors,
        "simple_grade_regression": simple,
        "grade_regression": regression,
        "prediction": prediction,
        "sensitivity_evaluated": sensitivity,
        "seed": SEED,
        "software": {
            "matplotlib": matplotlib.__version__,
            "numpy": np.__version__,
            "scipy": scipy.__version__,
        },
    }
    (ROOT / "results.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n"
    )
    if results_only:
        print(f"Analyzed {audit['records']:,} records. Wrote results.json.")
        return

    draw_figures(
        ROOT / "figures",
        result,
        outcomes,
        data[ADMISSION],
        data[semester(1, "approved")],
        residuals,
        curves,
    )
    print(f"Analyzed {audit['records']:,} records. Wrote results.json and figures/.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results-only", action="store_true", help="Write results.json without plots."
    )
    main(results_only=parser.parse_args().results_only)
