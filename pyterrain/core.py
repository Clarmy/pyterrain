import os
import shutil
from typing import List
from multiprocessing.dummy import Pool

import requests
import mercantile
import numpy as np
import pyproj

import matplotlib.pyplot as plt
from tqdm import tqdm
from retrying import retry


@retry
def single_download(arg):
    url, tmpfp, timeout = arg
    resp = requests.get(url, timeout=timeout, stream=True)
    if resp.ok:
        content = resp.content
    os.makedirs(os.path.dirname(tmpfp), exist_ok=True)
    try:
        with open(tmpfp, "wb") as f:
            f.write(content)
    except Exception:
        os.remove(tmpfp)
        return False
    else:
        return True


class Terrain:
    API_URL_PATTERN = "https://tile.nextzen.org/tilezen/terrain/v1/256/terrarium/{z}/{x}/{y}.png?api_key={api_key}"

    def __init__(self, api_key) -> None:
        self.api_key = api_key

    def _find_suitable_zoom(
        self,
        left: float,
        upper: float,
        right: float,
        lower: float,
        min_limit: int = 16,
        max_limit: int = 64,
    ) -> int:
        """To find a suitable zoom level for specific bound box.

        Args:
            left (float): Left longitude
            upper (float): Upper latitude
            right (float): Right longitude
            lower (float): Lower latitude
            min_limit (int, optional): The minimum zoom level limit. Defaults to 2.
            max_limit (int, optional): The maximum zoom level limit. Defaults to 6.

        Returns:
            int: The suitable zoom.
        """
        for zoom in range(15):
            left_upper_tile = mercantile.tile(left, upper, zoom)
            right_upper_tile = mercantile.tile(right, upper, zoom)
            right_lower_tile = mercantile.tile(right, lower, zoom)

            x_span = int(np.abs(right_upper_tile.x - left_upper_tile.x) + 1)
            y_span = int(np.abs(right_lower_tile.y - right_upper_tile.y) + 1)

            tile_nums = x_span * y_span

            if min_limit <= tile_nums <= max_limit:
                break
        else:
            print(f"tile_nums: {tile_nums}")

        return zoom

    def fetch(
        self,
        bbox: List,
        zoom: int = None,
        timeout=10,
        cache_path="./cache",
        keep_cache=True,
        coord="xy",
        multiproc=4,
        quiet=False,
    ) -> tuple:
        if quiet:
            progress_bar = False
        else:
            progress_bar = True

        left, upper, right, lower = bbox
        if zoom is None:
            zoom = self._find_suitable_zoom(left, upper, right, lower)

        left_upper_tile = mercantile.tile(left, upper, zoom)
        right_upper_tile = mercantile.tile(right, upper, zoom)
        right_lower_tile = mercantile.tile(right, lower, zoom)

        tile_left_x = left_upper_tile.x
        tile_right_x = right_upper_tile.x
        tile_upper_y = left_upper_tile.y
        tile_lower_y = right_lower_tile.y

        left_upper_bound = mercantile.bounds(left_upper_tile)
        right_lower_bound = mercantile.bounds(right_lower_tile)

        left_x, upper_y = mercantile.xy(left_upper_bound.west, left_upper_bound.north)
        right_x, lower_y = mercantile.xy(
            right_lower_bound.east, right_lower_bound.south
        )

        user_left_x, user_upper_y = mercantile.xy(*bbox[:2])
        user_right_x, user_lower_y = mercantile.xy(*bbox[2:])

        urls = []
        for nx, x in enumerate(range(tile_left_x, tile_right_x + 1, 1)):
            for ny, y in enumerate(range(tile_upper_y, tile_lower_y + 1, 1)):
                url = self.API_URL_PATTERN.format(
                    z=zoom, x=x, y=y, api_key=self.api_key
                )
                urls.append(((nx, ny), (x, y), url))

        shape = (
            256 * len(range(tile_upper_y, tile_lower_y + 1, 1)),
            256 * len(range(tile_left_x, tile_right_x + 1, 1)),
            4,
        )
        canvas = np.full(shape, np.nan)
        ys1d = np.linspace(upper_y, lower_y, shape[0])
        xs1d = np.linspace(left_x, right_x, shape[1])

        y_upper_diff = np.abs(ys1d - user_upper_y)
        y_lower_diff = np.abs(ys1d - user_lower_y)
        x_left_diff = np.abs(xs1d - user_left_x)
        x_right_diff = np.abs(xs1d - user_right_x)

        y_upper_i = np.where(y_upper_diff == y_upper_diff.min())[0][0]
        y_lower_i = np.where(y_lower_diff == y_lower_diff.min())[0][0]
        x_left_i = np.where(x_left_diff == x_left_diff.min())[0][0]
        x_right_i = np.where(x_right_diff == x_right_diff.min())[0][0]

        idx = np.s_[y_upper_i:y_lower_i, x_left_i:x_right_i]

        ys = (np.full(shape[:2], 1) * ys1d[np.newaxis].T)[idx]
        xs = (np.full(shape[:2], 1) * xs1d[np.newaxis])[idx]

        task_args = []
        for (nx, ny), (x, y), url in urls:
            tmpfp = os.path.join(cache_path, f"{zoom}/{x}/{y}.bin")
            if not os.path.exists(tmpfp):
                task_args.append([url, tmpfp, timeout])

        if task_args:
            with Pool(multiproc) as p:
                result = list(
                    tqdm(
                        p.imap_unordered(single_download, task_args),
                        total=len(task_args),
                        desc="downloading",
                    ),
                )

            try:
                assert all(result)
            except AssertionError:
                raise Exception("Not download completely, please retry")

        for (nx, ny), (x, y), url in tqdm(
            urls, disable=not progress_bar, desc="mosaicing"
        ):
            tmpfp = os.path.join(cache_path, f"{zoom}/{x}/{y}.bin")
            with open(tmpfp, "rb") as f:
                img = (plt.imread(f) * 255).astype(int)

            canvas[ny * 256 : ny * 256 + 256, nx * 256 : nx * 256 + 256] = img  # noqa

        red = canvas[..., 0]
        green = canvas[..., 1]
        blue = canvas[..., 2]
        elevation = ((red * 256 + green + blue / 256) - 32768).astype(int)[idx]

        if not quiet:
            print(f"bbox: {bbox}")
            print(f"zoom: {zoom}")
            print(f"mean of elevation: {int(elevation.mean())}")
            print(f"max of elevation: {elevation.max()}")
            print(f"min of elevation: {elevation.min()}")

        if not keep_cache:
            shutil.rmtree(cache_path)

        if coord == "lonlat":
            proj = pyproj.Transformer.from_crs(3857, 4326, always_xy=True)
            lons, lats = proj.transform(xs, ys)
            return lons, lats, elevation
        elif coord == "xy":
            return xs, ys, elevation


if __name__ == "__main__":
    pass
