# pyterrain
This is a Python package designed for terrain data fetching.

# Installation
To install the package, simply use pip: `$ pip install pyterrain`


# Usage

## Data Source
Pyterrain fetches Terrarium PNG tiles from the AWS Open Data elevation tiles source:
`https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{z}/{x}/{y}.png`


## Fetch DEM by bound box
You can download DEM data directly as follows:

```python
bbox = 116.972979,36.387619, 117.208694,36.172087  # This represents the Mount Taishan of China

terrain = Terrain()
xs, ys, elevation = terrain.fetch(bbox=bbox, quiet=False, coord="lonlat", zoom=10)
```

In case the download isn't completed due to connectivity issues, please retry the process.

## AI Skill

`pyterrain` supports installing AI Skill guidance so coding agents can use `pyterrain` APIs faster and with fewer mistakes.

After installing `pyterrain`, you can install skill files for different assistants:

```bash
pyterrain install-skill codex --mode local
pyterrain install-skill cursor --mode local
pyterrain install-skill claudecode --mode local
```

To install globally under your user home:

```bash
pyterrain install-skill codex --mode global
pyterrain install-skill cursor --mode global
pyterrain install-skill claudecode --mode global
```

Use `--dir /path/to/project` to choose another local workspace, and `--force` to overwrite an existing installation.
