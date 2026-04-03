# pyterrain API Cheatsheet

## Core API

### `Terrain.fetch(...)`

Signature:
`fetch(bbox, zoom=None, timeout=10, cache_path="./cache", keep_cache=True, coord="xy", multiproc=4, quiet=False)`

## Parameter intent

- `bbox`: target area in `[left, upper, right, lower]` order.
- `zoom`:
  - `None` for package auto-selection.
  - integer for fixed output resolution.
- `coord`:
  - `"xy"` for projected Web Mercator x/y.
  - `"lonlat"` for geographic longitude/latitude.
- `timeout`: per-request timeout for tile download.
- `cache_path`: tile cache directory.
- `keep_cache`: delete cache directory after run when `False`.
- `multiproc`: concurrent download workers.
- `quiet`: progress and stats output control.

## Return mapping

- `coord="xy"` -> `(xs, ys, elevation)`
- `coord="lonlat"` -> `(lons, lats, elevation)`

`elevation` is decoded integer-meter data from Terrarium tiles.

## Fast decision rules

- User needs geographic plotting / geospatial alignment:
  - choose `coord="lonlat"`.
- User needs projected-distance logic:
  - choose `coord="xy"`.
- User needs reproducible output:
  - set explicit `zoom` and stable `cache_path`.
- User has partial network failures:
  - keep cache, rerun, and increase `timeout` if needed.
