import os
from pathlib import Path

import matplotlib
import numpy as np
from matplotlib import font_manager
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy import stats

matplotlib.use("Agg")

OUTCOMES = ("Dropout", "Enrolled", "Graduate")
BLUE = "#7C9FC1"
ROSE = "#D8908E"
GREEN = "#83AA9D"
PURPLE = "#A99AC6"
OUTCOME_COLORS = (ROSE, PURPLE, GREEN)
INK = "#303844"
GRID = "#E5E9ED"
LOADINGS = LinearSegmentedColormap.from_list("loadings", [ROSE, "white", GREEN])


def configure_style():
    try:
        font_manager.findfont("Libertinus Serif", fallback_to_default=False)
    except ValueError:
        path = os.environ.get("LIBERTINUS_FONT")
        if path is None:
            cache = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
            roots = (
                cache / "Tectonic",
                Path.home() / "Library/Caches/TectonicProject.Tectonic",
            )
            path = next(
                (
                    font_path
                    for root in roots
                    for font_path in root.glob(
                        "bundles/data/*/LibertinusSerif-Regular.otf"
                    )
                ),
                None,
            )
        if path is None:
            raise RuntimeError(
                "Install Libertinus Serif or set LIBERTINUS_FONT to its regular OTF"
            )
        font_manager.fontManager.addfont(path)

    plt.rcParams.update(
        {
            "font.family": "Libertinus Serif",
            "font.size": 8.5,
            "axes.titlesize": 11,
            "axes.titleweight": "normal",
            "axes.titlecolor": INK,
            "axes.titlepad": 6,
            "axes.labelsize": 9.5,
            "axes.labelcolor": INK,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.edgecolor": INK,
            "axes.linewidth": 0.7,
            "axes.axisbelow": True,
            "grid.color": GRID,
            "grid.linewidth": 0.65,
            "xtick.color": INK,
            "ytick.color": INK,
            "xtick.major.size": 3,
            "ytick.major.size": 3,
            "text.color": INK,
            "legend.fontsize": 8,
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "pdf.compression": 9,
        }
    )


def save_figure(fig, folder, name):
    fig.savefig(
        folder / f"{name}.pdf",
        facecolor="white",
        metadata={"Creator": "MATH F432 project analysis", "CreationDate": None},
    )
    plt.close(fig)


def outcome_profiles(folder, result, outcomes, grades, passed):
    fig, (share, admission, completion) = plt.subplots(
        1, 3, figsize=(7.10, 2.30), layout="constrained", width_ratios=[0.95, 1, 1.12]
    )
    profiles = result["profiles"]
    for y, (label, color) in enumerate(zip(OUTCOMES, OUTCOME_COLORS)):
        profile = profiles[label]
        rate = 100 * profile["share"]
        low, high = np.array(profile["share_ci"]) * 100
        share.errorbar(
            rate,
            y,
            xerr=[[rate - low], [high - rate]],
            fmt="o",
            color=color,
            capsize=2,
            markersize=4,
            zorder=3,
        )
        share.text(high + 2, y, f"{rate:.1f}%", fontsize=7, va="center")
    share.set(
        yticks=range(3),
        yticklabels=OUTCOMES,
        ylim=(2.55, -0.55),
        xlim=(0, 65),
        xticks=[0, 20, 40, 60],
        xlabel="Share of records (%)",
    )
    share.tick_params(axis="y", length=0)
    share.grid(axis="x")
    share.set_title("Outcomes in the cohort")

    groups = [grades[outcomes == label] for label in OUTCOMES]
    boxes = admission.boxplot(
        groups,
        orientation="horizontal",
        positions=range(3),
        patch_artist=True,
        showfliers=False,
        widths=0.45,
        whis=(5, 95),
        medianprops={"color": INK, "linewidth": 1.1},
        whiskerprops={"color": INK, "linewidth": 0.8},
        capprops={"color": INK, "linewidth": 0.8},
    )
    for box, color in zip(boxes["boxes"], OUTCOME_COLORS):
        box.set_facecolor(color)
        box.set_edgecolor(color)
        box.set_alpha(0.55)
    admission.set(
        ylim=(2.55, -0.55),
        yticks=range(3),
        yticklabels=[""] * 3,
        xlabel="Admission grade (out of 200)",
    )
    admission.tick_params(axis="y", length=0)
    admission.grid(axis="x")
    admission.set_title("Admission grades")

    for label, color, style in zip(OUTCOMES, OUTCOME_COLORS, ("-", "--", ":")):
        completion.ecdf(
            passed[outcomes == label],
            color=color,
            ls=style,
            label=label,
            linewidth=1.55,
        )
    completion.set(
        xlabel="First-semester units passed",
        ylabel="Cumulative share",
        xlim=(-0.3, np.max(passed) + 0.5),
        ylim=(0, 1.04),
        yticks=[0, 0.5, 1],
        xticks=[0, 5, 10, 15, 20, 25],
    )
    completion.grid()
    completion.legend(loc="lower right", handlelength=1.65)
    completion.set_title("Units passed by outcome")
    save_figure(fig, folder, "outcome_profiles")


def classification(folder, result, curves):
    fig, (roc, odds) = plt.subplots(
        1, 2, figsize=(7.10, 2.50), layout="constrained", width_ratios=[1, 1.06]
    )
    roc.plot([0, 1], [0, 1], color=INK, ls=":", lw=0.9)
    prediction = result["prediction"]
    for name, (fpr, tpr) in curves.items():
        method, timing = name.split()
        color = BLUE if timing == "A" else GREEN
        style = "-" if method == "Logistic" else "--"
        value = prediction["models"][name]["auc"]
        roc.plot(
            fpr, tpr, ls=style, color=color, lw=1.45, label=f"{name} ({value:.3f})"
        )
    roc.set(
        xlim=(0, 1),
        ylim=(0, 1.02),
        xlabel="False positive rate",
        ylabel="Dropout recall",
        xticks=np.linspace(0, 1, 6),
        yticks=np.linspace(0, 1, 6),
    )
    roc.grid()
    roc.legend(loc="lower right", handlelength=1.7)
    roc.set_title("Identifying dropouts on the test set")

    terms = {
        "Age (5 years)": "Age, +5 years",
        "Admission grade (10 points)": "Admission grade, +10",
        "First-semester units enrolled": "Units enrolled, +1",
        "First-semester units passed": "Units passed, +1 (passers)",
        "First-semester grade among passers": "Grade among passers, +1",
    }
    table = prediction["models"]["Logistic B"]["odds_ratios"]
    odds.axvline(1, color=INK, ls=":", lw=0.9)
    for y, name in enumerate(terms):
        estimate = table[name]["estimate"]
        low, high = table[name]["ci"]
        odds.errorbar(
            estimate,
            y,
            xerr=[[estimate - low], [high - estimate]],
            fmt="o",
            color=BLUE if y < 2 else GREEN,
            capsize=2,
            markersize=4,
        )
    odds.set(
        xscale="log",
        yticks=range(len(terms)),
        yticklabels=list(terms.values()),
        ylim=(len(terms) - 0.45, -0.55),
        xlabel="Adjusted odds ratio for dropout",
    )
    ticks = np.array([0.25, 0.50, 1.00, 2.00, 3.00])
    odds.set_xticks(ticks, [f"{value:g}" for value in ticks])
    odds.minorticks_off()
    odds.tick_params(axis="y", length=0)
    odds.grid(axis="x")
    odds.set_title("Adjusted odds of dropout")
    save_figure(fig, folder, "classification")


def diagnostics(folder, outcomes, grades, residuals):
    fig, (entry, qq, variance) = plt.subplots(
        1, 3, figsize=(7.10, 2.17), layout="constrained"
    )
    for label, color, style in zip(OUTCOMES, OUTCOME_COLORS, ("-", "--", ":")):
        sample = grades[outcomes == label]
        sample = (sample - sample.mean()) / sample.std(ddof=1)
        theoretical, observed = stats.probplot(sample, fit=False)
        entry.plot(theoretical, observed, color=color, label=label, ls=style, lw=1.1)
    entry.plot([-3.7, 3.7], [-3.7, 3.7], color=INK, lw=0.9, alpha=0.6)
    entry.set(xlabel="Normal quantile", ylabel="Standardized grade")
    entry.legend(loc="upper left", handlelength=1.4)
    entry.set_title("Admission grade normality")

    fitted, residual = residuals
    (theoretical, observed), (slope, intercept, _) = stats.probplot(residual)
    qq.plot(theoretical, slope * theoretical + intercept, color=INK, ls="--", lw=0.9)
    qq.plot(theoretical, observed, color=BLUE, lw=1.2)
    qq.set(xlabel="Normal quantile", ylabel="Residual (grade points)")
    qq.set_title("Grade residual normality")

    variance.scatter(fitted, residual, s=3, color=BLUE, linewidths=0, alpha=0.24)
    variance.axhline(0, color=INK, ls="--", lw=0.9)
    variance.set(xlabel="Fitted grade (out of 20)", ylabel="Residual (grade points)")
    variance.set_title("Residuals by fitted grade")
    for ax in (entry, qq, variance):
        ax.grid()
    save_figure(fig, folder, "diagnostics")


def factor_structure(folder, factors):
    fig, (scree, loadings) = plt.subplots(
        1, 2, figsize=(7.10, 2.36), layout="constrained", width_ratios=[1, 0.95]
    )
    positions = np.arange(1, len(factors["labels"]) + 1)
    scree.plot(
        positions,
        factors["observed_eigenvalues"],
        "o-",
        color=BLUE,
        markersize=3.7,
        lw=1.4,
        label="Observed",
    )
    scree.plot(
        positions,
        factors["null_eigenvalues_p95"],
        "s--",
        color=INK,
        markersize=3.2,
        lw=1,
        label="Null 95th percentile",
    )
    scree.set(
        xticks=positions,
        xlabel="Ordered component",
        ylabel="Correlation eigenvalue",
        ylim=(0, None),
    )
    scree.grid()
    scree.legend(loc="upper right")
    scree.set_title("Choosing the number of factors")
    names = [name.replace("approved", "passed") for name in factors["labels"]]
    if "loadings" in factors:
        values = np.array(factors["loadings"])
        xlabels = [f"Factor {i}" for i in range(1, values.shape[1] + 1)]
        title = "How the measures load"
    else:
        values = np.array(factors["correlations"])
        xlabels = list(map(str, positions))
        title = "Academic correlations"
    x_edges = np.arange(values.shape[1] + 1) - 0.5
    y_edges = np.arange(values.shape[0] + 1) - 0.5
    loadings.pcolormesh(
        x_edges,
        y_edges,
        values,
        cmap=LOADINGS,
        vmin=-1,
        vmax=1,
        edgecolors="white",
        linewidth=0.6,
    )
    loadings.invert_yaxis()
    for (y, x), value in np.ndenumerate(values):
        loadings.text(x, y, f"{value:.2f}", ha="center", va="center", fontsize=8.5)
    loadings.set(
        yticks=range(len(names)),
        yticklabels=names,
        xticks=range(len(xlabels)),
        xticklabels=xlabels,
    )
    loadings.tick_params(length=0)
    loadings.set_title(title)
    for spine in loadings.spines.values():
        spine.set_visible(False)
    save_figure(fig, folder, "factor_structure")


def draw_figures(folder, result, outcomes, grades, passed, residuals, curves):
    configure_style()
    folder.mkdir(exist_ok=True)
    outcome_profiles(folder, result, outcomes, grades, passed)
    classification(folder, result, curves)
    diagnostics(folder, outcomes, grades, residuals)
    factor_structure(folder, result["factor_analysis"])
