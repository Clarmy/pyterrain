---
name: pyterrain
description: Use when writing, reviewing, or explaining Python code that uses pyterrain for terrain/elevation/DEM fetching via Terrarium tiles, including bbox-driven tile coverage, zoom and cache strategy, coordinate output mode selection (xy vs lonlat), and downstream integration with NumPy/matplotlib/rasterio/xarray workflows.
---

# pyterrain

## Overview

Use this skill when an AI needs to help with `pyterrain` code. The goal is to choose correct `Terrain.fetch(...)` parameters, avoid bbox/coordinate mistakes, and produce runnable snippets that fit terrain-fetch workflows.

For API behavior and parameter semantics, read [references/api-overview.md](references/api-overview.md).  
For practical task flow, read [references/workflows.md](references/workflows.md).  
For debugging and anti-hallucination checks, read [references/common-pitfalls.md](references/common-pitfalls.md).

## When To Use

- The user mentions `pyterrain`.
- The user asks for DEM/elevation retrieval in a specific area.
- The task includes bounding box tile fetch/mosaic logic.
- The user asks about `zoom`, `coord`, `cache_path`, `keep_cache`, `multiproc`, or timeout tuning.
- The user needs to plug fetched terrain arrays into plotting or geospatial analysis code.

## Recommended Execution Order

- Confirm bbox ordering is `[left, upper, right, lower]`.
- Choose `coord` according to downstream consumer expectations.
- Start with `zoom=None` (auto) unless fixed resolution is required.
- Keep cache during iteration unless cleanup is explicitly needed.
- If failures occur, check network/download integrity before changing numerical logic.

## Current Behavior You Must Respect

- Entry API is `Terrain().fetch(...)`.
- Data source is Terrarium PNG tiles from AWS open elevation tiles.
- Returned tuple is always coordinate grid arrays plus elevation array.
- `coord="xy"` returns Web Mercator x/y coordinates.
- `coord="lonlat"` returns geographic longitude/latitude coordinates.
- `keep_cache=False` removes the cache directory after fetch.
- The decoded `elevation` output is integer meters.

## Capability Boundaries

- `pyterrain` handles terrain tile fetch, mosaic, and decode.
- CRS-heavy transformations, advanced raster IO, and geospatial file writing typically belong to downstream libraries (for example rasterio/xarray/geopandas).
- Do not describe `pyterrain` as a full GIS processing framework.

## Coding Style For AI-Generated Examples

- Prefer complete snippets with imports, bbox definition, and one `fetch(...)` call.
- Explain why chosen `coord` mode matches the target workflow.
- Keep examples minimal and runnable before adding optimization complexity.
- For debugging snippets, prefer `quiet=False` to expose fetch diagnostics.

## Common Mistakes To Avoid

- Swapping bbox upper/lower latitude order.
- Misinterpreting projected x/y outputs as lon/lat.
- Choosing overly large zoom/bbox combinations without considering tile count.
- Deleting cache unintentionally during iterative development.
- Guessing undocumented APIs instead of using known `Terrain.fetch(...)` behavior.
