# API Cheatsheet

## Import

```python
from pyterrain import Terrain
```

## Minimal fetch

```python
terrain = Terrain()
xs, ys, elev = terrain.fetch(
    bbox=[left, upper, right, lower],
)
```

## Parameter quick guide

- `bbox`:
  - Required order: `[left, upper, right, lower]`
  - Interpreted as lon/lat degrees
- `zoom`:
  - `None` lets package auto-pick a reasonable range
  - Set explicit zoom for reproducible resolution
- `coord`:
  - `"xy"` -> Web Mercator projected grid
  - `"lonlat"` -> longitude/latitude grid
- `cache_path`:
  - Directory for downloaded tile binaries
- `keep_cache`:
  - `True` keeps downloaded tiles for reuse
  - `False` deletes the entire cache directory after fetch
- `timeout`:
  - Request timeout for each tile download
- `multiproc`:
  - Worker count for parallel tile downloads
- `quiet`:
  - `False` prints bbox/zoom/elevation diagnostics

## Return semantics

- `coord="xy"` -> `(xs, ys, elevation)`
- `coord="lonlat"` -> `(lons, lats, elevation)`
- `elevation` unit is integer meters decoded from Terrarium tiles

## Common decision patterns

- Need geographic plotting or geospatial alignment:
  - Use `coord="lonlat"`
- Need projected-distance-style computation:
  - Use `coord="xy"`
- Need stable repeated runs:
  - Fix `zoom`
  - Use project-local `cache_path`
  - Keep `keep_cache=True`
- Network not stable:
  - Increase `timeout`
  - Keep cache and rerun to finish missing tiles
