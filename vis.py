from tiatoolbox.visualization.tileserver import TileServer
from tiatoolbox.wsicore.wsireader import OpenSlideWSIReader, WSIReader

# # wsi = WSIReader.open("/home/john/Desktop/test1.tiff")
# wsi = WSIReader.open("/home/john/Downloads/test1.jp2")

# app = TileServer(
#     title="Testing TileServer",
#     layers={
#         "Test file": wsi,
#     },
# )

# app.run()


wsi = OpenSlideWSIReader("/home/john/Desktop/test1.tiff", mpp=0.5)

# Read a region in the middle and plot it
from matplotlib import pyplot as plt

region = wsi.read_bounds((35_000, 40_000, 45_000, 50_000), resolution=16, units="mpp")

plt.imshow(region)
plt.show()
