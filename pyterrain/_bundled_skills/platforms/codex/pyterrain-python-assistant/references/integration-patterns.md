# Integration Patterns

## Pattern 1: Fetch then immediate plotting

1. Query with `coord="lonlat"` for map-like axes that expect geographic coordinates.
2. Plot elevation with `matplotlib` (and `cartopy` if projection-aware map rendering is needed).
3. Keep `quiet=False` during initial tuning.

## Pattern 2: Fetch for numerical analysis

1. Query with `coord="xy"` when projected coordinates are preferred.
2. Feed arrays into NumPy/xarray calculations.
3. Export or persist using raster-focused tooling (for example rasterio/rioxarray).

## Pattern 3: Stable pipeline runs

1. Fix `zoom` after initial calibration.
2. Use project-local `cache_path`.
3. Keep `keep_cache=True` for iterative jobs.
4. Add explicit retry strategy around batch processing scripts.

## Pattern 4: Robust network behavior

1. Detect partial failures as download integrity issues first.
2. Retry with the same cache to avoid re-downloading completed tiles.
3. Increase `timeout` before changing computational code.
