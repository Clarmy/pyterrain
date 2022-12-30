import io
from typing import List

import requests
import mercantile
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


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
            int: The suitable zoome.
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

    def fetch(self, bbox: List, zoom: int = None, progress_bar=False) -> np.ndarray:
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

        urls = []
        for nx, x in enumerate(range(tile_left_x, tile_right_x + 1, 1)):
            for ny, y in enumerate(range(tile_upper_y, tile_lower_y + 1, 1)):
                url = self.API_URL_PATTERN.format(
                    z=zoom, x=x, y=y, api_key=self.api_key
                )
                urls.append(((nx, ny), url))

        shape = (
            256 * len(range(tile_upper_y, tile_lower_y + 1, 1)),
            256 * len(range(tile_left_x, tile_right_x + 1, 1)),
            4,
        )
        canvas = np.full(shape, np.nan)

        for (nx, ny), url in tqdm(urls, disable=not progress_bar):
            resp = requests.get(url, timeout=7, stream=True)
            if resp.ok:
                buffer = io.BytesIO(resp.content)
                img = (plt.imread(buffer) * 255).astype(int)
                canvas[ny * 256 : ny * 256 + 256, nx * 256 : nx * 256 + 256] = img

        red = canvas[..., 0]
        green = canvas[..., 1]
        blue = canvas[..., 2]
        elevation = ((red * 256 + green + blue / 256) - 32768).astype(int)

        return elevation


if __name__ == "__main__":
    bbox = 76.98806, 37.454188, 103.2594, 19.70498

    terrain = Terrain("Dto0r88DQuaQizoxcQScvw")

    elevation = terrain.fetch(bbox=bbox, progress_bar=True)

    print(f"max of elevation: {elevation.max()}")
    fig = plt.figure(
        figsize=(elevation.shape[1] / 100, elevation.shape[0] / 100), dpi=100
    )
    ax = plt.Axes(fig, [0.0, 0.0, 1.0, 1.0])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.imshow(elevation, cmap=plt.cm.terrain)
    fig.savefig("./tibat-plateau.png")
