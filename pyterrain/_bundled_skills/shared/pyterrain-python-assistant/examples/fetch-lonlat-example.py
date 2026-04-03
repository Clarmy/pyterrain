from pyterrain import Terrain


bbox = [116.972979, 36.387619, 117.208694, 36.172087]
terrain = Terrain()
lons, lats, elevation = terrain.fetch(bbox=bbox, zoom=10, coord="lonlat", quiet=False)

print(lons.shape, lats.shape, elevation.shape)
