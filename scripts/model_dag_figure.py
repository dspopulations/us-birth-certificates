"""Render the DSP004 dependency graph with its planned extension points.

Produces ``notes/figures/dsp004_dag_extensions`` (PNG + SVG). The upper panel is
the generative structure of the preferred accounting model in mathematical
notation; the lower panel lists the five extension points marked on it, in the
dependency order recommended by the model-family review.

No data or fit artefacts are read -- this is a structural diagram only.

Usage:
    python scripts/model_dag_figure.py
"""

from __future__ import annotations  # noqa: I001

import dspopulations_us_birth_certificates.env_guard  # noqa: F401

import os  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402
from dse_research_utils.environment import setup  # noqa: E402
from dse_research_utils.plot import styles  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

from dspopulations_us_birth_certificates.plot_utils import save_fig  # noqa: E402

OUTPUT_DIR = "notes/figures"
FILE_NAME = "dsp004_dag_extensions"

# Node roles. Fixed external inputs and observed data are not estimated; free
# parameters are the only two quantities DSP004 samples.
ROLE_STYLE = {
    "data": {"fc": "#e8e8e8", "ec": "#5a5a5a"},
    "fixed": {"fc": "#fdece0", "ec": styles.COLOUR_DARK_ORANGE},
    "free": {"fc": "#e3edf8", "ec": styles.COLOUR_DARK_BLUE},
    "derived": {"fc": "#ffffff", "ec": "#8a8a8a"},
    "estimand": {"fc": "#e6f2e8", "ec": styles.COLOUR_DARK_GREEN},
}
BADGE_COLOUR = styles.COLOUR_DARK_PURPLE

# (key, x, y, width, height, role, title, detail)
NODES = [
    ("N", 3.0, 44.0, 15.0, 11.0, "data", r"$N_{y,a}$", "livebirths\n(observed)"),
    ("theta", 21.0, 82.0, 21.0, 12.0, "fixed", r"$\theta_a$", "Morris curve, fixed"),
    (
        "rho",
        46.0,
        82.0,
        26.0,
        12.0,
        "free",
        r"$\mathrm{logit}\,\rho_y \sim"
        r" \mathcal{N}(\mathrm{logit}\,r_y,\ \sigma_y)$",
        "surveillance prior, 9 free",
    ),
    ("s", 76.0, 82.0, 21.0, 12.0, "free", r"$s$", "weak prior, 1 free"),
    ("eta", 46.0, 64.0, 26.0, 10.0, "derived", r"$\eta_y = 1 - \rho_y$", ""),
    (
        "pds",
        24.0,
        45.0,
        38.0,
        11.0,
        "derived",
        r"$p^{\mathrm{DS}}_{y,a} = \theta_a\,\eta_y$",
        "",
    ),
    (
        "T",
        68.0,
        45.0,
        29.0,
        11.0,
        "estimand",
        r"$T = \sum_{y,a} N_{y,a}\,\theta_a\,\eta_y$",
        "the estimand",
    ),
    ("f", 80.0, 26.0, 17.0, 10.0, "fixed", r"$f$", "fixed at $7.8\\times10^{-5}$"),
    (
        "prec",
        20.0,
        26.0,
        54.0,
        10.0,
        "derived",
        r"$p^{\mathrm{rec}}_{y,a} = p^{\mathrm{DS}}_{y,a}\,s"
        r" + (1 - p^{\mathrm{DS}}_{y,a})\,f$",
        "",
    ),
    (
        "R",
        24.0,
        6.0,
        46.0,
        11.0,
        "data",
        r"$R_{y,a} \sim \mathrm{Binomial}(N_{y,a},\ p^{\mathrm{rec}}_{y,a})$",
        "the only observation",
    ),
]

# (from, to, from_side, to_side)
EDGES = [
    ("theta", "pds", "bottom", "top"),
    ("rho", "eta", "bottom", "top"),
    ("eta", "pds", "bottom", "top"),
    ("pds", "T", "right", "left"),
    ("pds", "prec", "bottom", "top"),
    ("s", "prec", "bottom", "top"),
    ("f", "prec", "left", "right"),
    ("prec", "R", "bottom", "top"),
    ("N", "pds", "right", "left"),
    ("N", "R", "bottom", "left"),
]

# (label, x, y) in axes data units. Placed in gaps rather than offset from a
# node corner, so a badge never lands on a box edge or an arrow.
BADGES = [
    ("1", 22.5, 97.0),
    ("2", 55.0, 78.0),
    ("3", 77.5, 97.0),
    ("4", 26.5, 20.5),
    ("5", 75.5, 67.0),
]

EXTENSIONS = [
    (
        "1",
        "Assisted reproduction on the natural rate",
        r"$\theta_a \;\rightarrow\; \theta_a\,\kappa^{\mathrm{ART}}$"
        r"$\quad$ for $\mathtt{rf\_artec}=Y$",
        "Donor oocytes and embryo screening make the Morris curve wrong for "
        "ART births.\nExplains 93% of the 45+ age misfit. Do this first.",
    ),
    (
        "2",
        "Group and age structure on reduction",
        r"$\rho_y \;\rightarrow\; \rho_{g,y}"
        r" + \beta_g\,x_a \quad$ with $\sum_g w_g\,\rho_{g,y} = \rho_y$",
        "The payer-by-age gradient identifies $\\beta_g$; recording cannot "
        "produce it.\nMargin constraint preserves the national annual anchor.",
    ),
    (
        "3",
        "Group offsets on recording",
        r"$s \;\rightarrow\; s_g = \mathrm{logit}^{-1}"
        r"(\sigma_0 + \sigma_g), \quad \sum_g \sigma_g = 0$",
        "Time-invariant by assumption. Identifies group contrasts in recording, "
        "not levels;\nthe level stays confounded with $\\rho$.",
    ),
    (
        "4",
        "Confirmed and pending as separate channels",
        r"$R \;\rightarrow\; (R^{C}, R^{P})$ with $(s_C, f_C)$ and $(s_P, f_P)$",
        "Doubles the constraints to 702 for two extra parameters and lets the "
        "data estimate $f$,\nwhich is enriched in the pending channel by about "
        "3.4 times.",
    ),
    (
        "5",
        "Clinical severity on recording only",
        r"$s_g \;\rightarrow\; s_g\,h(m), \quad"
        r" m \perp \theta_a, \eta_y \mid D$",
        "Severity enters the recording channel alone, so it supplies variation "
        "in $s$ that does\nnot covary with $\\theta$. This is what pins the "
        "level extension 3 leaves free.",
    ),
]


def _anchor(node: tuple, side: str) -> tuple[float, float]:
    """Return the (x, y) edge-attachment point on one side of a node box."""
    _, x, y, w, h = node[:5]
    return {
        "top": (x + w / 2.0, y + h),
        "bottom": (x + w / 2.0, y),
        "left": (x, y + h / 2.0),
        "right": (x + w, y + h / 2.0),
    }[side]


def _badge(ax: plt.Axes, x: float, y: float, label: str, size: float = 8.0) -> None:
    """Draw a numbered extension badge.

    Rendered as a text bounding box rather than a ``Circle`` patch: the axes are
    not square, so a patch radius in data units would come out elliptical.
    """
    ax.text(
        x,
        y,
        label,
        ha="center",
        va="center",
        fontsize=size,
        color="white",
        fontweight="bold",
        zorder=5,
        clip_on=False,
        bbox={
            "boxstyle": "circle,pad=0.34",
            "facecolor": BADGE_COLOUR,
            "edgecolor": "white",
            "linewidth": 1.1,
        },
    )


def draw_dag(ax: plt.Axes) -> None:
    """Draw the DSP004 generative structure with extension badges."""
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 108)
    ax.axis("off")
    ax.text(
        50,
        105,
        "DSP004 accounting model: dependency structure and planned extensions",
        ha="center",
        va="center",
        fontsize=12,
    )
    by_key = {node[0]: node for node in NODES}

    for _key, x, y, w, h, role, title, detail in NODES:
        style = ROLE_STYLE[role]
        ax.add_patch(
            FancyBboxPatch(
                (x, y),
                w,
                h,
                boxstyle="round,pad=0,rounding_size=1.4",
                linewidth=1.4,
                facecolor=style["fc"],
                edgecolor=style["ec"],
                zorder=2,
            )
        )
        title_y = y + h * 0.62 if detail else y + h / 2.0
        ax.text(
            x + w / 2.0,
            title_y,
            title,
            ha="center",
            va="center",
            fontsize=11,
            zorder=3,
        )
        if detail:
            ax.text(
                x + w / 2.0,
                y + h * 0.24,
                detail,
                ha="center",
                va="center",
                fontsize=7.5,
                color="#4a4a4a",
                zorder=3,
            )

    for src, dst, src_side, dst_side in EDGES:
        start = _anchor(by_key[src], src_side)
        end = _anchor(by_key[dst], dst_side)
        connection = (
            "arc3,rad=0"
            if src_side in {"top", "bottom"}
            else "angle3,angleA=0,angleB=90"
        )
        ax.add_patch(
            FancyArrowPatch(
                start,
                end,
                arrowstyle="-|>",
                mutation_scale=11,
                linewidth=1.1,
                color="#6a6a6a",
                connectionstyle=connection,
                shrinkA=3,
                shrinkB=4,
                zorder=1,
            )
        )

    for label, x, y in BADGES:
        _badge(ax, x, y, label)

    handles = [
        FancyBboxPatch(
            (0, 0),
            1,
            1,
            boxstyle="round,pad=0",
            facecolor=ROLE_STYLE[role]["fc"],
            edgecolor=ROLE_STYLE[role]["ec"],
            linewidth=1.3,
        )
        for role in ("data", "fixed", "free", "derived", "estimand")
    ]
    ax.legend(
        handles,
        (
            "observed / given",
            "fixed external input",
            "free parameter (2 in total)",
            "deterministic",
            "estimand",
        ),
        loc="lower right",
        bbox_to_anchor=(1.0, -0.02),
        fontsize=7.5,
        frameon=False,
        ncol=1,
        handlelength=1.1,
        handleheight=1.1,
        borderpad=0.2,
        labelspacing=0.45,
    )


def draw_extensions(ax: plt.Axes) -> None:
    """List the numbered extension points beneath the graph."""
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")
    ax.text(
        0,
        97,
        "Extension points, in dependency order",
        fontsize=9.5,
        fontweight="bold",
        va="top",
    )
    for idx, (label, title, maths, detail) in enumerate(EXTENSIONS):
        top = 89.0 - idx * 19.0
        _badge(ax, 2.2, top - 2.0, label, size=7.5)
        ax.text(6.5, top + 0.6, title, fontsize=9, fontweight="bold", va="top")
        ax.text(6.5, top - 5.4, maths, fontsize=9.5, va="top")
        ax.text(
            53.0,
            top + 0.4,
            detail,
            fontsize=8,
            color="#3f3f3f",
            va="top",
            linespacing=1.5,
        )


def main() -> int:
    setup.init_script()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    fig, (ax_dag, ax_ext) = plt.subplots(
        2,
        1,
        figsize=(12.0, 11.5),
        gridspec_kw={"height_ratios": [1.5, 1.0], "hspace": 0.02},
    )
    draw_dag(ax_dag)
    draw_extensions(ax_ext)
    save_fig(fig, OUTPUT_DIR, FILE_NAME, dpi=styles.DPI_FILE)
    plt.close(fig)
    print(f"wrote {FILE_NAME} (png/svg) to {OUTPUT_DIR}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
