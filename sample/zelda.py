import copy

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors

from pyterrain import Terrain

if __name__ == "__main__":

    bbox = 108.444319, 20.161757, 111.318897, 18.05883  # 海南岛

    terrain = Terrain("qBD4m7PNT5apV-Xl7PROxA")

    xs, ys, elevation = terrain.fetch(bbox=bbox, progress_bar=True, coord="lonlat", zoom=12)

    land = copy.deepcopy(elevation)
    land[land < 0] = -9999

    fig = plt.figure(
        figsize=(elevation.shape[1] / 100, elevation.shape[0] / 100), dpi=100
    )
    ax = plt.Axes(fig, [0.0, 0.0, 1.0, 1.0])
    ax.set_axis_off()
    fig.add_axes(ax)

    hyrule = colors.LinearSegmentedColormap.from_list(
        "hyrule", ["#3D2E00", "#C6C7B0"]
    )  # colormap for hyrule land
    ax.contourf(xs, ys, land, cmap=hyrule, levels=np.arange(5, land.max(), 2), zorder=3)

    ax.contourf(
        xs, ys, elevation, levels=[elevation.min(), 0], colors=["#212A2D"], zorder=1
    )

    ax.contourf(xs, ys, elevation, levels=[0, 5], colors=["#41535A"], zorder=4)

    ax.contour(
        xs,
        ys,
        elevation,
        colors="#382D06",
        levels=np.arange(5, elevation.max(), 20),
        alpha=0.6,
        linewidths=0.4,
        zorder=4,
    )

    ax.contour(
        xs,
        ys,
        elevation,
        colors="#382D06",
        levels=np.arange(5, elevation.max(), 100),
        alpha=0.6,
        linewidths=0.8,
        zorder=5,
    )
    fig.savefig("./hynanrule.png")
