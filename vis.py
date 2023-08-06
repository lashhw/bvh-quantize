import pyvista as pv
import numpy as np
import sys

if len(sys.argv) != 2:
    print('usage: python vis.py MODEL_FILE')

ply_mesh = pv.read(sys.argv[1])
p = pv.Plotter()

cluster_size = np.fromfile('cluster_size.bin', np.uint32)
cluster_sum = np.r_[0, np.cumsum(cluster_size)]
bbox = np.fromfile('bbox.bin', np.float32).reshape(-1, 6)

cluster_idx = 0
def click_callback(pos):
    global cluster_idx
    p.clear()
    p.add_mesh(ply_mesh, style='wireframe')
    box_mesh = pv.PolyData()
    for i in range(cluster_sum[cluster_idx], cluster_sum[cluster_idx + 1]):
        box_mesh += pv.Box(bbox[i])
    p.add_mesh(box_mesh, color='red', style='wireframe')
    cluster_idx += 1

p.track_click_position(click_callback)
p.add_mesh(ply_mesh, opacity=0.1)
p.show()
