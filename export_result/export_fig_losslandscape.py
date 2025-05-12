import pickle
import os
import sys
sys.path.append('/home/xinmeng/pinn_bow_mass')
from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import scipy
matplotlib.use('module://backend_interagg')
from matplotlib.ticker import ScalarFormatter
# import tikzplotlib
## fb = 100

def load_data(path):
    with open(path, "rb") as f:
        data = pickle.load(f)
    A = data[0]
    B = data[1]
    loss_grid = data[2]
    # loss_grid = np.log10(np.abs(loss_grid))
    return A, B, loss_grid





fb = 10
type = 'pinn' # 'deep'
if type == 'pinn':
    nn_num =1
    path = f"/home/xinmeng/pinn_bow_mass/trained_model/fa25/export_figure/losslandscape_data/losslandscape_{fb}_pinn_nn{nn_num}.pkl"
elif type == 'deep':
    path = f"/home/xinmeng/pinn_bow_mass/trained_model/fa25/export_figure/losslandscape_data/losslandscape_{fb}_deeponet.pkl"
A, B, loss_grid= load_data(path)
#
data_to_save = {
    'A': A,
    'B': B ,
    'loss_grid':loss_grid
}


scipy.io.savemat(f'/home/xinmeng/pinn_bow_mass/trained_model/fa25/export_figure/losslandscape_data/mat/losslandscape_{fb}_{type}.mat', data_to_save)

print('')

import matplotlib.ticker as mticker

# My axis should display 10⁻¹ but you can switch to e-notation 1.00e+01
# def log_tick_formatter(val, pos=None):
#     return f"$10^{{{int(val)}}}$"  # remove int() if you don't use MaxNLocator
#     # return f"{10**val:.2e}"      # e-Notation
#
# from matplotlib.colors import Normalize



fig = plt.figure(figsize=(5.5*2, 3.5*2))
font = 20
ax = fig.add_subplot(111, projection='3d')
# z_min, z_max = np.min(loss_grid), np.max(loss_grid)

# norm = Normalize(vmin=z_min, vmax=z_max)
# Plot surface with manually set norm
surf = ax.plot_surface(A, B, np.log10(loss_grid), cmap="viridis", shade=False)

cbar = fig.colorbar(surf, ax=ax)
# cbar.ax.yaxis.set_major_formatter(mticker.FuncFormatter(log_tick_formatter))

cbar.set_label("Loss", fontsize=font)
cbar.ax.tick_params(labelsize=font)
ax.set_xlabel(r"$\varepsilon_1$", fontsize=font, labelpad=10)
ax.set_ylabel(r"$\varepsilon_2$", fontsize=font, labelpad=10)
# ax.set_zlim(z_min, z_max)

ax.set_title(r"$F_B = {}$".format(fb), fontsize=font)
# ax.zaxis.set_major_formatter(mticker.FuncFormatter(log_tick_formatter))
# ax.zaxis.set_major_locator(mticker.MaxNLocator(integer=True))
ax.tick_params(axis='x', labelsize=font)
ax.tick_params(axis='y', labelsize=font)
# ax.tick_params(axis='z', which='both', bottom=False, top=False, labelbottom=False)  # Hide z-axis ticks and labels
ax.tick_params(axis='z', labelsize=font)
ax.grid(False)
ax.view_init(elev=0, azim=45)
fig.tight_layout()

if type  == 'pinn':
    plt.savefig(
        f'/home/xinmeng/pinn_bow_mass/trained_model/fa25/export_figure/losslandscape_pinn_{fb}_nn{nn_num}.pdf')
elif type == 'deep':
    plt.savefig(f"/home/xinmeng/pinn_bow_mass/trained_model/fa25/export_figure/losslandscape_{fb}_deeponet.pdf")
plt.show()
print('')
