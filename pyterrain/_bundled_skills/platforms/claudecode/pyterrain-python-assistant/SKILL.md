# pyterrain Python Assistant

Use this skill for `pyterrain` data-fetch and DEM workflows.

Checklist:
- Validate bbox order is `[left, upper, right, lower]`
- Prefer auto zoom first (`zoom=None`)
- Use `coord="lonlat"` when integrating with geographic workflows
- Keep cache unless explicit cleanup is required

References:
- `references/api-overview.md`
- `references/workflows.md`
- `references/common-pitfalls.md`
