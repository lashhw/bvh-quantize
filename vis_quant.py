import hashlib
import numpy as np
import pyvista as pv
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

sw = 0.125
num_samples = 2048
num_colors = 20

print('creating grid...')
u = np.linspace(0, 2 * np.pi, num_samples)
v = np.linspace(0, np.pi, num_samples)
x = np.outer(np.cos(u), np.sin(v))
y = np.outer(np.sin(u), np.sin(v))
z = np.outer(np.ones(np.size(u)), np.cos(v))
grid = pv.StructuredGrid(x, y, z)

def summary(a, b, c):
    def transform(x):
        if x >= (1 << 29):
            return 0b100000000
        if x <= -(1 << 30):
            return 0b100000001
        if x < -128:
            return 0b110000000 | int(x).bit_length()
        if x > 127:
            return 0b111000000 | int(x).bit_length()
        return 0b11111111 & x
    encoded = transform(a) | (transform(b) << 9) | (transform(c) << 18)
    in_range = lambda x: -128 <= x and x <= 127
    quant_type = 1 if in_range(a) and in_range(b) and in_range(c) else 0
    return encoded, quant_type

def hash_to_color(encoded, quant_type):
    x_bytes = int(encoded).to_bytes(32, 'big')
    h = hashlib.sha256(x_bytes).digest()
    c = int.from_bytes(h, 'big') % num_colors
    if quant_type == 0:
        c += num_colors
    return c

print('computing color...')
cx = np.empty(grid.n_cells, dtype=float)
cy = np.empty(grid.n_cells, dtype=float)
cz = np.empty(grid.n_cells, dtype=float)
for i in range(grid.n_cells):
    cx[i], cy[i], cz[i] = grid.get_cell(i).center
ix = np.floor(1.0 / (cx * sw)).astype(np.int64)
iy = np.floor(1.0 / (cy * sw)).astype(np.int64)
iz = np.floor(1.0 / (cz * sw)).astype(np.int64)
colors = np.zeros(grid.n_cells, dtype=int)
for i in range(grid.n_cells):
    encoded, quant_type = summary(ix[i], iy[i], iz[i])
    colors[i] = hash_to_color(encoded, quant_type)
grid.cell_data['color'] = colors

print('plotting...')
p = pv.Plotter()
cmap1 = plt.get_cmap('tab20')
cmap2 = plt.get_cmap('tab20b')
combined_cmap = ListedColormap(cmap1.colors + cmap2.colors)
p.add_mesh(grid, cmap=combined_cmap)
p.show_grid()
p.remove_scalar_bar()
p.show()
