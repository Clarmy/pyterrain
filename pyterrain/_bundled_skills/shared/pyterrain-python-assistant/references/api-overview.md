# API Overview

## Main entry
`from pyterrain import Terrain`

## `Terrain.fetch(...)`

Signature:
`fetch(bbox, zoom=None, timeout=10, cache_path="./cache", keep_cache=True, coord="xy", multiproc=4, quiet=False)`

Key parameters:
- `bbox`: `[left, upper, right, lower]`, longitudes/latitudes in WGS84.
- `zoom`: tile zoom. If `None`, a suitable zoom is auto-selected.
- `coord`: `"xy"` (Web Mercator) or `"lonlat"` (geographic).
- `keep_cache`: remove cache if `False`.
- `quiet`: print useful diagnostics when `False`.

Return value:
- `(xs, ys, elevation)` for `coord="xy"`
- `(lons, lats, elevation)` for `coord="lonlat"`

`elevation` values are decoded from Terrarium tiles and returned as integer meters.
