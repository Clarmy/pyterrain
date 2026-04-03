# Common Pitfalls

- Bbox order is wrong.
  `pyterrain` expects `[left, upper, right, lower]`, not `[left, lower, right, upper]`.

- Coordinate interpretation mismatch.
  `coord="xy"` returns Web Mercator x/y values, not lon/lat.

- Cache cleanup surprises.
  Setting `keep_cache=False` deletes the whole `cache_path` directory after fetch.

- Large areas with high zoom.
  Tile count grows quickly. Start with `zoom=None` or lower zoom levels.

- Silent failures in unstable networks.
  Use non-quiet mode and retry if downloads are incomplete.
