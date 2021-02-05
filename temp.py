from pycwr.io import read_auto
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from pycwr.draw.RadarPlot import GraphMap,Graph
import numpy as np
# file = r"D:\20180609\Z_RADR_I_Z9250_20190701003800_O_DOR_SA_CAP.bin.bz2"
# PRD = read_auto(file)
# PyartRadar = PRD.ToPyartRadar()
# ax = plt.axes(projection=ccrs.PlateCarree())
# graph = GraphMap(PRD,ccrs.PlateCarree())
# graph.plot_crf_map(ax, 0, "dBZ", cmap="pyart_NWSRef")
# ax.set_title("example of CAPPI with map", fontsize=16)
# plt.show()


# def bz2_to_numpy(file):
#     PRD = read_auto(file)
#     fig, ax = plt.subplots()
#     graph = Graph(PRD)
#     graph.plot_crf(ax)
#     product = graph.Radar.product
#     CR = product['CR'].values
#     CR = np.where(np.isnan(CR), 0, CR)
#     return CR
#
# data = bz2_to_numpy(r"D:\20180609\Z_RADR_I_Z9250_20180609025700_O_DOR_SA_CAP.bin.bz2")

import os
import numpy as np
import cv2
from utils import gray2RGB

new_test_dir = r'D:\datan\huanan\huanan\test2'

for root,dirs,files in os.walk(r'D:\datan\huanan\huanan\test'):
    for file in files:
        if os.path.exists(new_test_dir + '\\'+file) is False:
            os.makedirs(new_test_dir + '\\'+file)
        data = np.load(root+"\\"+file)
        data = data/10
        data[data>70]=70
        data[data<0]=0
        data = np.uint8(data)

        for i in range(data.shape[0]):
            cv2.imwrite(new_test_dir + '\\'+file+"\\"+str(i)+".jpg",gray2RGB(data[i,0]))

