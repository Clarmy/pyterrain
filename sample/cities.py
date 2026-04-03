import copy

import numpy as np
import matplotlib.pyplot as plt

# import matplotlib.colors as colors
import cartopy.crs as ccrs
from matplotlib.colors import LightSource
from cnmaps import get_adm_maps, draw_map
from pyterrain import Terrain

if __name__ == "__main__":

    # bbox = 108.444319, 20.161757, 111.318897, 18.05883  # 海南岛
    # bboxs = {
    #     '南京市':(118.538536, 32.210923, 118.987545,31.936777),
    #     '上海市':(),
    #     '北京市':(),
    #     '西安市':(),
    #     '成都市':(),
    #     '重庆市':()
    # }

    cities = ["北京市", "上海市", "西安市", "成都市"]

    for city in cities:
        mappolygon = get_adm_maps(city=city, record="first", only_polygon=True)
        left, right, lower, upper = mappolygon.get_extent(buffer=0)

        bbox = (left, upper, right, lower)

        print("bbox", bbox)

        terrain = Terrain()

        lon, lat, dem = terrain.fetch(bbox=bbox, quiet=False, coord="lonlat", zoom=12)

        dem = dem.astype(np.float32)
        print("dem", dem)
        dem[dem < 0] = -9999

        ls = LightSource(azdeg=360, altdeg=30)

        rgb = ls.shade(
            dem,
            cmap=plt.cm.gist_earth,
            blend_mode="overlay",
            vert_exag=0.5,
            dx=10,
            dy=10,
            fraction=1.5,
            vmin=0,
        )

        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection=ccrs.PlateCarree())

        img = ax.imshow(
            rgb,
            extent=(lon.min(), lon.max(), lat.min(), lat.max()),
            transform=ccrs.PlateCarree(),
        )

        map_polygon = get_adm_maps(country="中华人民共和国", record="first", only_polygon=True)

        draw_map(mappolygon, color="w", linewidth=2)

        ax.set_extent((left, right, lower, upper))

        fig.savefig(f"./{city}.png", bbox_inches="tight", pad_inches=0, dpi=300)

    # land = copy.deepcopy(elevation)
    # land[land < 0] = -9999

    # fig = plt.figure(
    #     figsize=(elevation.shape[1] / 100, elevation.shape[0] / 100), dpi=100
    # )
    # ax = plt.Axes(fig, [0.0, 0.0, 1.0, 1.0])
    # ax.set_axis_off()
    # fig.add_axes(ax)

    # hyrule = colors.LinearSegmentedColormap.from_list(
    #     "hyrule", ["#3D2E00", "#C6C7B0"]
    # )  # colormap for hyrule land
    # ax.contourf(xs, ys, land, cmap=hyrule, levels=np.arange(5, land.max(), 2), zorder=3)

    # ax.contourf(
    #     xs, ys, elevation, levels=[elevation.min(), 0], colors=["#212A2D"], zorder=1
    # )

    # ax.contourf(xs, ys, elevation, levels=[0, 5], colors=["#41535A"], zorder=4)

    # ax.contour(
    #     xs,
    #     ys,
    #     elevation,
    #     colors="#382D06",
    #     levels=np.arange(5, elevation.max(), 20),
    #     alpha=0.6,
    #     linewidths=0.4,
    #     zorder=4,
    # )

    # ax.contour(
    #     xs,
    #     ys,
    #     elevation,
    #     colors="#382D06",
    #     levels=np.arange(5, elevation.max(), 100),
    #     alpha=0.6,
    #     linewidths=0.8,
    #     zorder=5,
    # )
    # fig.savefig("./hyrule-nanjing.png")
