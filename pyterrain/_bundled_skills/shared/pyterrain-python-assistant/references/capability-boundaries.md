# Capability Boundaries

## What pyterrain is responsible for

- Determine tile coverage from bbox and zoom strategy.
- Download Terrarium tiles from the configured endpoint.
- Mosaic downloaded tiles into one raster canvas.
- Decode Terrarium RGB values into elevation meters.
- Return coordinate grids plus elevation array.

## What pyterrain is not designed to fully handle

- Rich vector GIS operations (overlay, dissolve, topology editing).
- Full raster data engineering workflows (window IO, pyramids, tiled GeoTIFF writing).
- CRS-heavy reprojection pipelines between many coordinate systems.
- Cartographic styling systems beyond simple fetch and numerical outputs.

## Typical downstream libraries by responsibility

- `matplotlib` / `cartopy`:
  - Map rendering and projection-aware plotting.
- `xarray` / `numpy`:
  - Numerical analysis, multidimensional processing, resampling logic.
- `rasterio` / `rioxarray`:
  - Geospatial raster IO, CRS metadata, export pipelines.
- `geopandas` / `shapely`:
  - Vector boundary operations and joins.

## Integration guidance for AI responses

- Use `pyterrain` for data acquisition and initial terrain arrays.
- Hand off advanced visualization and storage to dedicated downstream tools.
- If user asks for end-to-end GIS export, explain which part is `pyterrain` and which part needs other libraries.
- Do not claim `pyterrain` alone can replace a full GIS stack.
