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
