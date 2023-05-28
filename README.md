# pyterrain
This is a Python package designed for terrain data fetching.

# Installation
To install the package, simply use pip: `$ pip install pyterrain`


# Usage

## Register API Key
Pyterrain itself does not provide terrain data. Instead, it retrieves data from an external source. In order to download this data, you first need to register for an API key at [nextzen.org](https://developers.nextzen.org/). An example of the key format would be `Dto0r88DQuaQizoxcQSxxx`.


## Fetch DEM by bound box
Once your API key is set up, you can proceed to download DEM data as follows:

```python
bbox = 116.972979,36.387619, 117.208694,36.172087  # This represents the Mount Taishan of China

terrain = Terrain("Dto0r88DQuaQizoxcQSxxx")  # Insert your API key
xs, ys, elevation = terrain.fetch(bbox=bbox, quiet=False, coord="lonlat", zoom=10)
```

In case the download isn't completed due to connectivity issues, please retry the process.
