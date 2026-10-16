"""Palmer penguins, four ways -- the reference raincloud and three variations."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")

from matplotlib.colors import to_hex

import numpy as np
import pandas as pd
import seaborn as sns

from raincloudkit import BoxSpec, GroupStyle, RainStyle, plot_raincloud

HERE = Path(__file__).parent
penguins = sns.load_dataset("penguins").dropna(subset=["body_mass_g", "flipper_length_mm"])

# 1. One pooled row. The rain carries no second variable, so the dots are free
#    to be translucent and let overlap read as density.
grey = RainStyle(
    show_total=True,
    group_tick_rotation=90,
    value_label="Body Mass (g)",
    cloud_facecolor="#808080", cloud_edgecolor="black", cloud_alpha=1.0, cloud_linewidth=2.2,
    box_facecolor="#808080", box_edgecolor="black", box_linewidth=2.2,
    point_color="#3C6E9F", point_edgecolor="none", point_alpha=0.45, point_size=55,
    value_limits=(2000, 7000), value_ticks=range(2000, 7001, 1000),
    figsize=(9, 5), axis_linewidth=2.0, tick_labelsize=20, axis_label_fontsize=20,
    cloud_base=0.14, cloud_width=0.46, rain_offset=-0.07, rain_width=0.13, box_offset=-0.36,
)
plot_raincloud(penguins, "body_mass_g", style=grey).savefig(HERE / "1_penguins_total.png")

# 1b. The same distribution at k sample sizes, 10 ... 10^k, smallest on top.
#     Penguins has only 342 rows, so every row is a bootstrap draw (with
#     replacement) from the observed masses -- only n changes between rows.
K = 4
sizes = [10 ** i for i in range(1, K + 1)]
labels = [f"n = {n:,}" for n in sizes]
rng = np.random.default_rng(0)
masses = penguins["body_mass_g"].to_numpy()
samples = pd.DataFrame({
    "body_mass_g": np.concatenate([rng.choice(masses, n, replace=True) for n in sizes]),
    "n": np.repeat(labels, sizes),
})
# one alpha cannot serve 10 dots and 10^k dots: dense rows get smaller, fainter
# points so the rain keeps showing density instead of filling in solid
per_row = {
    label: GroupStyle(point_size=55 * (10 / n) ** 0.3, point_alpha=min(0.6, 1.2 / n ** 0.3))
    for label, n in zip(labels, sizes)
}
stacked = grey.derive(
    show_total=False, group_order=labels, group_styles=per_row,
    group_tick_rotation=0, group_label="", figsize=(9, 2.2 * K),
    # 1's geometry spans more than one slot; stacked rows would collide
    cloud_width=0.34, box_offset=-0.32,
)
plot_raincloud(samples, "body_mass_g", "n", stacked).savefig(HERE / "1_penguins_total_var_2.png")

# 1c. 1b on its side: n = 10 ... 10^k from left to right.
plot_raincloud(
    samples, "body_mass_g", "n", stacked.derive(orientation="vertical", figsize=(2.6 * K, 8)),
).savefig(HERE / "1_penguins_total_var_3.png")

# 1d. 1b regrouped by layer: every cloud overlaid, then all rain rows, then all
#     box rows. The overlaid clouds need telling apart, so each n gets its own
#     grey -- light for n = 10, dark for 10^k -- and its box shares that shade,
#     which makes the box band double as the key.
shades = [to_hex(c) for c in matplotlib.colormaps["Greys"](np.linspace(0.4, 0.95, K))]
per_layer = {
    label: GroupStyle(point_size=row.point_size, point_alpha=row.point_alpha,
                      cloud_facecolor=shade, cloud_edgecolor=shade, box_facecolor=shade)
    for (label, row), shade in zip(per_row.items(), shades)
}
plot_raincloud(
    samples, "body_mass_g", "n",
    stacked.derive(layout="layered", group_styles=per_layer, figsize=None, slot_size=1.5,
                   cloud_width=1.4, cloud_alpha=0.12, cloud_linewidth=2.6,
                   # with every cloud on one baseline, the closing edges are just clutter
                   cloud_close_outline=False),
).savefig(HERE / "1_penguins_total_var_4.png")

# 2. One row per species, mean +/- std instead of median/IQR.
species = plot_raincloud(
    penguins, "body_mass_g", "species",
    RainStyle(value_label="Body mass (g)", group_label="", title="Body mass by species",
              annotate=True, annotate_format="{center:.0f} g  (n={n})",
              box_facecolor="white", cloud_alpha=0.5),
    BoxSpec(center="mean", box="std", whisker="1.5iqr"),
)
species.savefig(HERE / "2_penguins_species.png")
print(species.stats.to_string(index=False))

# 3. Vertical, with a categorical point hue and a per-group override.
plot_raincloud(
    penguins, "flipper_length_mm", "species",
    RainStyle(orientation="vertical", value_label="Flipper length (mm)", group_label="Species",
              show_total=True, total_position="first", jitter="normal", legend_title="Sex",
              grid=True, point_size=18,
              group_styles={"Total": GroupStyle(color="#9E9E9E", label="All penguins")}),
    point_hue="sex",
).savefig(HERE / "3_penguins_vertical.png")

# 4. A bare mapping -- no DataFrame, no columns.
plot_raincloud(
    {sp: g["bill_length_mm"].dropna() for sp, g in penguins.groupby("species", observed=True)},
    style=RainStyle(value_label="Bill length (mm)", show_box=False, cloud_cut=1.0),
).savefig(HERE / "4_penguins_mapping.png")
print("wrote 4 figures")
