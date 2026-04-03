---
name: pyterrain-python-assistant
description: Use when the user is writing or debugging Python code that fetches elevation/DEM data with pyterrain, especially Terrarium tile download and mosaicing workflows, bbox-based terrain extraction, coordinate output selection (Web Mercator xy vs lon/lat), cache and retry behavior, or integration with NumPy, matplotlib, rasterio, xarray, and GIS-style post-processing.
---

# pyterrain Skill Overview

Use this skill when helping with Python code that uses `pyterrain` to fetch DEM/elevation from Terrarium tiles, decode elevations, and integrate the resulting grids into geospatial or scientific workflows.

## When to Use This Skill

- The user mentions `pyterrain` directly.
- The user asks to fetch terrain/elevation/DEM by a bounding box.
- The user discusses Terrarium tiles, map tile mosaicing, or tile download retries.
- The user needs help choosing `coord="xy"` versus `coord="lonlat"`.
- The user wants to post-process fetched elevation with NumPy/matplotlib/rasterio/xarray.
- The user sees unexpected shapes, coordinate orientation, or cache behavior after `fetch(...)`.

## What Cursor Agent Should Do

- First classify the task as: data fetch, coordinate interpretation, cache/retry reliability, or downstream integration.
- Treat `pyterrain` as a Python library API (`Terrain().fetch(...)`), not as a standalone data server.
- Confirm bbox order is `[left, upper, right, lower]` before debugging anything else.
- Clarify coordinate semantics early:
  - `coord="xy"` returns Web Mercator x/y grid.
  - `coord="lonlat"` returns longitude/latitude grid.
- For unstable network cases, explain retry and cache usage clearly before suggesting major refactors.
- Prefer runnable code that includes imports, bbox, and a complete `Terrain().fetch(...)` call.

## pyterrain Knowledge Model

`pyterrain` downloads Terrarium PNG tiles from the AWS elevation tile endpoint, mosaics tiles into one canvas, decodes elevation, and returns `(x_like, y_like, elevation)` arrays. It is commonly used as a lightweight terrain-fetch layer before visualization or raster analysis in other packages.

## How to Work with Supporting Files

- Read `references/api-overview.md` for parameter and return-value semantics.
- Read `references/api-cheatsheet.md` for quick API and parameter selection.
- Read `references/workflows.md` for recommended usage patterns.
- Read `references/capability-boundaries.md` to separate pyterrain responsibilities from GIS toolchain responsibilities.
- Read `references/common-pitfalls.md` to avoid common bbox/coordinate/cache mistakes.
- Prefer `examples/` when writing runnable snippets.

## Output Requirements

- Prefer directly runnable Python code or minimal-diff edits.
- Explain why selected fetch parameters (`zoom`, `coord`, `cache_path`, `keep_cache`) fit the user task.
- Include downstream handling when relevant (for example visualization or export preparation), not only a single API call.
- Do not invent unverified APIs or CLI commands.

## Common Failure Modes

- Bounding box order is inverted.
- Treating `coord="xy"` output as lon/lat degrees.
- Using too high zoom for a large bbox and causing large tile counts.
- Deleting useful cache unexpectedly with `keep_cache=False`.
- Assuming failures are algorithmic when they are partial-download/network issues.
