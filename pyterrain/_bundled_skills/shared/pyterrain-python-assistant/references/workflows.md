# Workflows

## 1. Fetch DEM for a study area
1. Define bbox as `[left, upper, right, lower]`.
2. Instantiate `Terrain()`.
3. Call `fetch(..., coord="lonlat")` when plotting with lon/lat-aware tools.

## 2. Keep runs reproducible
- Set explicit `zoom` once you find the desired resolution.
- Provide a project-local `cache_path` for repeated runs.
- Keep `keep_cache=True` for faster reruns.

## 3. Integrate with numerical workflows
- Use `coord="xy"` if downstream operations assume projected meters.
- Use `coord="lonlat"` for geospatial libraries expecting lon/lat grids.
