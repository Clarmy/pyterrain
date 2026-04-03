---
name: pyterrain-python-assistant
description: Use when the user is writing, reviewing, or debugging Python workflows with pyterrain to fetch terrain/DEM data from Terrarium tiles, including bbox query design, zoom choice, coordinate output mode (xy or lonlat), cache/retry behavior, and integration with plotting or geospatial post-processing libraries.
---

# pyterrain Skill Overview

Use this guidance when the task involves `pyterrain`-based elevation fetching, tile mosaicing, terrain array interpretation, and the handoff to downstream scientific or mapping code.

## When to Use This Skill

- The user explicitly asks about `pyterrain` APIs.
- The task asks for DEM or elevation extraction by area/bbox.
- The conversation mentions terrain tiles, Terrarium PNG decoding, or elevation array generation.
- The user needs help understanding `zoom`, `coord`, `cache_path`, or `keep_cache`.
- The user is integrating fetched arrays into matplotlib/NumPy/rasterio/xarray pipelines.

## What Claude Should Do

- Start by validating task intent: fetch-only, visualization, or analysis pipeline.
- Verify bbox order and coordinate assumptions first; these are the most common root causes.
- Prefer real `Terrain.fetch(...)` behavior over guessed parameter semantics.
- Be explicit about coordinate outputs:
  - `coord="xy"`: Web Mercator projected coordinates.
  - `coord="lonlat"`: geographic longitudes and latitudes.
- If reliability is the issue, focus on timeout/retry/cache strategy before suggesting unrelated code changes.
- If environment execution is not possible, still provide correct runnable snippets and explain expected outputs.

## pyterrain Knowledge Model

Treat `pyterrain` as a focused Python terrain-fetch utility: it computes tile coverage from a bbox, downloads Terrarium tiles, mosaics them, decodes elevations, and returns coordinate/elevation arrays. It does not replace full GIS stacks; it is typically one stage in a broader workflow.

## How to Work with Supporting Files

- Use `references/api-overview.md` for API contracts.
- Use `references/workflows.md` for practical execution patterns.
- Use `references/common-pitfalls.md` for debugging-first checks.
- Use `examples/` to ground final code in runnable templates.

## Output Requirements

- Provide code that can run with minimal local edits.
- Justify parameter choices (`zoom`, `coord`, cache strategy) in plain terms.
- Include enough surrounding code to make integration clear.
- Avoid API hallucinations or undocumented behavior claims.

## Common Failure Modes

- Wrong bbox order (`left, lower, right, upper` used by mistake).
- Misreading projected x/y as lon/lat.
- Excessive tile requests due to bbox/zoom mismatch.
- Cache directory unexpectedly removed when `keep_cache=False`.
- Attributing download interruptions to logic bugs without checking network conditions.
