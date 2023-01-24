# pyterrain
A python package to fetch terrain.

# Installation
You can install by pip `$ pip install pyterrain`

# Usage

## Register API Key
Pyterrain don't offer terrain data itself, it fetches data from another website. Before downloading data, you should sign up an API key from [nextzen.org](https://developers.nextzen.org/). The key's pattern is like `Dto0r88DQuaQizoxcQSxxx`

## Fetch DEM by bound box
When API key is reday, you can download DEM data like this:

```python
bbox = 108.444319, 20.161757, 111.318897, 18.05883  # Hainan province of China

terrain = Terrain("Dto0r88DQuaQizoxcQSxxx")  # Pass API key
xs, ys, elevation = terrain.fetch(bbox=bbox, quiet=False, coord="lonlat", zoom=10)
```

If download is not complete because of connection, retry it.
