import numpy as np
import rasterio


file_name = "./data/test/LC81240322017118LGN00_B1.TIF"

# Read raster bands directly to Numpy arrays.
with rasterio.open(file_name) as src:
    r = src.read()
    print r.shape

exit();
# Combine arrays in place. Expecting that the sum will
# temporarily exceed the 8-bit integer range, initialize it as
# a 64-bit float (the numpy default) array. Adding other
# arrays to it in-place converts those arrays "up" and
# preserves the type of the total array.
total = np.zeros(r.shape)
for band in r, g, b:
    total += band
total /= 3

# Write the product as a raster band to a new 8-bit file. For
# the new file's profile, we start with the meta attributes of
# the source file, but then change the band count to 1, set the
# dtype to uint8, and specify LZW compression.
profile = src.profile
profile.update(dtype=rasterio.uint8, count=1, compress='lzw')

with rasterio.open('example-total.tif', 'w', **profile) as dst:
    dst.write(total.astype(rasterio.uint8), 1)

with rasterio.open('tests/data/RGB.byte.tif') as src:
    print(src.width, src.height)
    print(src.crs)
    print(src.transform)
    print(src.count)
    print(src.indexes)