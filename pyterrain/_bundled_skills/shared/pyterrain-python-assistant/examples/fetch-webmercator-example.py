from pyterrain import Terrain


bbox = [116.972979, 36.387619, 117.208694, 36.172087]
terrain = Terrain()
xs, ys, elevation = terrain.fetch(bbox=bbox, zoom=10, coord="xy", quiet=False)

print(xs.shape, ys.shape, elevation.shape)
