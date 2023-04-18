from pathlib import Path
from typing import Tuple

import numpy as np  # noqa: F401
import pandas as pd
from matplotlib import gridspec  # noqa: F401
from matplotlib import pyplot as plt

# # Use Latex to for text
# plt.rc("text", usetex=True)
# plt.rc("font", family="serif")

RUNS_DIR = Path("benchmark/runs")
RUN = "latest"

if RUN == "latest":
    runs = list(RUNS_DIR.iterdir())
    RUN = max(runs, key=lambda p: p.stat().st_mtime)

print(f"Using run: {RUN}")

PLOTS_DIR = RUN / "plots"
PLOTS_DIR.mkdir(exist_ok=True)


def format_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Format dataframe for plotting."""
    df = df.copy()
    # Remove any non NaN error rows except for "TileSizeError"
    df = df[(df["error"].isna()) | (df["error"] == "TileSizeError")]
    # Remove "wsic-repack" tool
    df = df[df["tool"] != "wsic-repack"]
    # Merge bfconvert and bioformats2raw tools
    df.loc[df["tool"] == "bioformats2raw", "tool"] = "bfconvert/bioformats2raw"
    df.loc[df["tool"] == "bfconvert", "tool"] = "bfconvert/bioformats2raw"
    df.reset_index(inplace=True, drop=True)
    df = df.rename(
        columns={
            "in_format": "From",
            "out_format": "To",
        }
    )
    df.set_index(["From", "To"], append=False, inplace=True)
    return df


def hprint(
    *strings: str,
    level: int = 1,
    wide=True,
    padding: Tuple[int, int] = (1, 0),
    sep="",
    end="\n",
    flush=False,
) -> None:
    """Print with a heading."""
    import os

    columns = os.get_terminal_size()[0] or 80
    width = columns if wide else sum(len(s) for s in strings)
    header_char = " ━─┄┈·"[level]
    print(
        (
            f"{end*padding[0]}"
            f"{header_char * width}"
            f"{sep.join(strings)}"
            f"{end}"
            f"{header_char * width}"
            f"{end*padding[1]}"
        ),
        sep=sep,
        end=end,
        flush=flush,
    )


results = pd.read_csv(RUN / "results.csv")
results = format_dataframe(results)
results["GP/s"] = results.gigapixels / results.time

# Pivot Table
pivot = results.pivot_table(
    # index=["From", "To"],
    # columns="tool",
    index="tool",
    columns=["From", "To"],
    values="GP/s",
    aggfunc="mean",
)

# ggplot style
plt.style.use("ggplot")

# Colour bars by the series / columns (from, to)
# Where the fill is from and to is the edge
FORMAT_COLORS = {
    "jp2": "cyan",
    "tiff": "red",
    "zarr": "magenta",
    "dcm": "navy",
    "svs": "green",
}
# The above colours are UGLY, let's make some nicer ones with RGB
FORMAT_COLORS = {
    "jp2": "#42c7f3",  # Offical JP2 Cyan
    "tiff": "#86A873",  #
    "zarr": "#e01073",  # Official Zarr magenta
    "dcm": "#0a1128",  # Oxford DCM blue
    "svs": "#BB9F06",
}


def cols2fill(cols: Tuple[str, str]) -> str:
    """Get the fill colour for a column."""
    from_, to_ = cols
    return FORMAT_COLORS.get(from_, "black")


def cols2edge(cols: Tuple[str, str]) -> str:
    """Get the edge colour for a column."""
    from_, to_ = cols
    return FORMAT_COLORS.get(to_, ("black"))


COLORS_DICT = {
    ("jp2", "dcm"): (0, 0, 1),
    # (jp2, jp2)
    ("jp2", "svs"): (0, 0, 0.6),
    ("jp2", "tiff"): (0, 0, 0.4),
    ("jp2", "zarr"): (0, 0, 0.2),
    ("svs", "dcm"): (0, 1, 0),
    ("svs", "jp2"): (0, 0.8, 0),
    # (svs, svs)
    ("svs", "tiff"): (0, 0.4, 0),
    ("svs", "zarr"): (0, 0.2, 0),
    ("tiff", "dcm"): (1, 0, 0),
    ("tiff", "jp2"): (0.8, 0, 0),
    ("tiff", "svs"): (0.6, 0, 0),
    # (tiff, tiff)
    ("tiff", "zarr"): (0.2, 0, 0),
    ("dcm", "jp2"): (1, 0, 1),
    ("dcm", "svs"): (0.6, 0, 0.6),
    ("dcm", "tiff"): (0.4, 0, 0.4),
    # (dcm, dcm)
    ("dcm", "zarr"): (0.2, 0, 0.2),
}

COLORS_DICT = {
    ("jp2", "dcm"): "#032c64",
    # (jp2, jp2)
    ("jp2", "svs"): "#2E7FBD",
    ("jp2", "tiff"): "#44C6F3",
    ("jp2", "zarr"): "#9EE6FF",
    ("svs", "dcm"): "#810000",
    ("svs", "jp2"): "#E61B1B",
    # (svs, svs)
    ("svs", "tiff"): "#FF5E00",
    ("svs", "zarr"): "#FF9F5D",
    ("tiff", "dcm"): "#4C005D",
    ("tiff", "jp2"): "#7E0462",
    ("tiff", "svs"): "#E01073",
    # (tiff, tiff)
    ("tiff", "zarr"): "#FF8BC2",
    ("dcm", "jp2"): "#357200",
    ("dcm", "svs"): "#6ABE22",
    ("dcm", "tiff"): "#9FFF00",
    # (dcm, dcm)
    ("dcm", "zarr"): "#D0FF81",
}


# Plot
GROUP_SPACING = 0
BAR_SPACING = 0
ax = pivot.plot.bar(
    figsize=(15, 4.5),
    width=(1 - GROUP_SPACING),
    color=COLORS_DICT,
    # color=pivot.columns.map(cols2fill),
    # hatch="//",
    linewidth=1,
    logy=True,
    grid=True,
    legend=True,
    title="Conversion Speed",
    rot=0,
    xlabel="Tool",
    ylabel="Gigapixels/Second",
)

# # Add bar labels
# for container, (index, series) in zip(ax.containers, pivot.items()):
#     from_, to_ = index
#     highest_value_position = series.argmax()
#     for i, bar in enumerate(container.patches):
#         x, y = bar.get_xy()
#         # Draw from -> to text
#         if bar.get_height() > 1e-4:
#             ax.text(
#                 x=x + bar.get_width() / 2,
#                 y=y + bar.get_height() / 2,
#                 # s=f"{from_} → {to_}{' *' if i == highest_value_position else ''}",
#                 s="*" if i == highest_value_position else "",
#                 va="bottom",
#                 ha="center",
#                 rotation=0,
#                 # color="white",
#                 color=cols2edge((from_, to_)),
#                 fontsize=10,
#                 weight=800,
#             )
#     # ax.bar_label(
#     #     container=container,
#     #     labels=[f"{from_} → {to_}"],
#     #     label_type="edge",
#     #     # Rotate the labels
#     #     rotation=90,
#     #     padding=-100,
#     # )

# Map legend labels to "From → To" format
legend_labels = [f"{from_} → {to_}" for from_, to_ in pivot.columns]

# Make a legend outside
legend = plt.legend(
    bbox_to_anchor=(1.05, 1), loc="upper left", title="From → To", labels=legend_labels
)

# Grid behind bars
plt.gca().set_axisbelow(True)

# Get the best value for each (from, to)
best_values = pivot.max(axis=0)


def outline_bars(pivot: pd.DataFrame) -> None:
    # Set the outline colour for each bar to the "to" format
    for (from_, to_), _ in pivot.items():
        # Iterate over the bars for this (from, to)
        for bar in ax.patches[
            pivot.columns.get_loc((from_, to_))
            * len(pivot) : (pivot.columns.get_loc((from_, to_)) + 1)
            * len(pivot)
        ]:
            # Set the edge colour
            bar.set_edgecolor(cols2edge((from_, to_)))
        # Set the legend edge colour also
        legend.get_patches()[pivot.columns.get_loc((from_, to_))].set_edgecolor(
            cols2edge((from_, to_))
        )


# Set the width of all bars (to add spacing within groups)
for bar in ax.patches:
    # Set bar width
    bar.set_width(((1 - BAR_SPACING) / len(pivot.columns)) * (1 - GROUP_SPACING))


def hatch_highest(pivot: pd.DataFrame, hatch: str = "//") -> None:
    """Find the hightest bar for each (from, to) and hatch it."""
    for from_, to_ in best_values:
        # Find the fastest tool for this (from, to)
        fastest_tool = pivot.idxmax(axis=0)[(from_, to_)]
        # Get the bar for this (from, to) and fastest tool
        bar = ax.patches[
            pivot.columns.get_loc((from_, to_)) * len(pivot)
            + pivot.index.get_loc(fastest_tool)
        ]
        # Hatch the bar
        bar.set_hatch(hatch)
        bar.set_edgecolor((1, 1, 1, 0.67))


hatch_highest(pivot)


def add_value_labels(pivot: pd.DataFrame) -> None:
    """Add value labels and * to highest value in each conversion.

    Iterate over all (from, to) conversions in the pivot table and add a
    label to the bar with the value of the fastest tool for that
    conversion. If a bar has a value above a certain minimum size, also
    add a label with the value of the bar. Finally, add a * to the label
    of the highest bar for each conversion.

    Args:
        pivot:
            A pandas DataFrame with (from, to) conversions as columns
            and tools as rows, where each cell contains the time for the
            tool to perform the conversion.

    Returns:
        None.
    """
    # Add value labels to all bars (above a minimum size)
    # and add a * to the highest value for each (from, to) conversion
    for (from_, to_), series in pivot.items():
        print(f"{from_} → {to_}")
        print("-" * 40)
        print(series)
        print("-" * 40)
        # Find the fastest tool for this (from, to)
        fastest_tool = pivot.idxmax(axis=0)[(from_, to_)]
        print("Fastest tool:", fastest_tool)
        print("-" * 40)
        # Get the bar for this (from, to) and fastest tool
        bar = ax.patches[
            pivot.columns.get_loc((from_, to_)) * len(pivot)
            + pivot.index.get_loc(fastest_tool)
        ]
        # Set the label
        bar.set_label(f"{series[fastest_tool]:.2f}")


plt.tight_layout()
plt.savefig(PLOTS_DIR / "bar_plot.png")
plt.savefig(PLOTS_DIR / "bar_plot.pdf")
