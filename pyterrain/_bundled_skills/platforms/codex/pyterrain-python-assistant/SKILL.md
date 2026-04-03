# pyterrain Python Assistant (Codex)

Use this skill when writing code that fetches terrain/elevation data with `pyterrain`.

## What `pyterrain` does
- Downloads Terrarium PNG elevation tiles from:
  `https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{z}/{x}/{y}.png`
- Mosaics tiles and decodes elevation into integer meters.
- Returns raster arrays in either:
  - Web Mercator (`coord="xy"`)
  - Lon/lat (`coord="lonlat"`)

## Preferred workflow
1. Construct a `Terrain()` object.
2. Call `fetch(...)` with a bbox in `[left, upper, right, lower]` order.
3. Start with `zoom=None` to auto-choose a reasonable tile range.
4. Use `coord="lonlat"` if downstream code expects geographic coordinates.
5. Keep `quiet=False` while debugging so tile and elevation stats are printed.

## Read next
- `references/api-overview.md`
- `references/workflows.md`
- `references/common-pitfalls.md`
- `examples/fetch-lonlat-example.py`
- `examples/fetch-webmercator-example.py`
