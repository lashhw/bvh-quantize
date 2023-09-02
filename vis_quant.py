import hashlib
import numpy as np
import pyvista as pv
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from tqdm import tqdm

num_samples = 2048
num_colors = 20

print('creating grid...')
u = np.linspace(0, 2 * np.pi, num_samples)
v = np.linspace(0, np.pi, num_samples)
x = np.outer(np.cos(u), np.sin(v))
y = np.outer(np.sin(u), np.sin(v))
z = np.outer(np.ones(np.size(u)), np.cos(v))
grid = pv.StructuredGrid(x, y, z)
del u, v, x, y, z

def summary(a, b, c):
    def transform(x):
        sign = 1 if x & (1 << 31) else 0
        exponent = (x >> 23) & 0b011111111
        mantissa = x & ((1 << 23) - 1)

        new_exponent = (exponent - (127 + 7)) & 0b11111
        new_mantissa = 0b10000000 | (mantissa >> 16)

        return ((sign << 13) | (new_exponent << 8) | new_mantissa), (new_exponent & 1)

    ta, ea = transform(a)
    tb, eb = transform(b)
    tc, ec = transform(c)
    encoded = (ta << 28) | (tb << 14) | tc
    return encoded, (ea ^ eb ^ ec)

def hash_to_color(encoded, quant_type):
    x_bytes = int(encoded).to_bytes(6, 'big')
    h = hashlib.sha256(x_bytes).digest()
    c = int.from_bytes(h, 'big') % num_colors
    if quant_type:
        c += num_colors
    return c

print('computing color...')
cx = np.empty(grid.n_cells, dtype=np.float32)
cy = np.empty(grid.n_cells, dtype=np.float32)
cz = np.empty(grid.n_cells, dtype=np.float32)
for i in tqdm(range(grid.n_cells)):
    cx[i], cy[i], cz[i] = grid.get_cell(i).center
sw = 0.0078125  # 2^(-7)
ix = np.floor(1.0 / (cx * sw)).view(np.uint32)
iy = np.floor(1.0 / (cy * sw)).view(np.uint32)
iz = np.floor(1.0 / (cz * sw)).view(np.uint32)
grid.cell_data['color'] = np.empty(grid.n_cells, dtype=int)
for i in tqdm(range(grid.n_cells)):
    encoded, quant_type = summary(ix[i], iy[i], iz[i])
    grid.cell_data['color'][i] = hash_to_color(encoded, quant_type)
del cx, cy, cz, ix, iy, iz

print('plotting...')
p = pv.Plotter()
cmap1 = plt.get_cmap('tab20b')
cmap2 = plt.get_cmap('tab20c')
combined_cmap = ListedColormap(cmap1.colors + cmap2.colors)
p.add_mesh(grid, cmap=combined_cmap)
p.show_grid()
p.remove_scalar_bar()
p.show()
