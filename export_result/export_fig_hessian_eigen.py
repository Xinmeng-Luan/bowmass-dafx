import pickle
import os
import sys
sys.path.append('/home/xinmeng/pinn_bow_mass')
from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt
import density_plot
# import tikzplotlib
import matplotlib
# matplotlib.use('module://backend_interagg')
def load_nn_result(path):
    with open(path, "rb") as f:
        data = pickle.load(f)
    density_eigen = data[0]
    density_weight = data[1]
    return density_eigen, density_weight

def load_fd_result(fb):
    p_q = loadmat(f'../data/p_q_fd_05_fb_{fb}.mat')

    t_p_fd = p_q['t_p'].squeeze()
    t_q_fd = p_q['t_q'].squeeze()
    p_fd = p_q['p'].squeeze()
    q_fd = p_q['q'].squeeze()
    return t_p_fd, t_q_fd, p_fd, q_fd

fb = 1000

deeponet_path = f"/home/xinmeng/pinn_bow_mass/trained_model/fa25/export_figure/hessian_eigen_data/hessian_eigen_fb_{fb}_deeponet.pkl"
pinn_path = f"/home/xinmeng/pinn_bow_mass/trained_model/fa25/export_figure/hessian_eigen_data/hessian_eigen_fb_{fb}_pinn_nn1.pkl"



density_eigen_deep, density_weight_deep = load_nn_result(deeponet_path)
density_eigen_pinn, density_weight_pinn = load_nn_result(pinn_path)
density_deep, grids_deep = density_plot.density_generate(density_eigen_deep, density_weight_deep, num_bins=2000000,overhead=0.01,sigma_squared=1e-4)
density_pinn, grids_pinn = density_plot.density_generate(density_eigen_pinn, density_weight_pinn, num_bins=2000000,overhead=0.01,sigma_squared=1e-4)

font = 25

fig, ax = plt.subplots(1, 1, figsize=(5.5*2, 3.5*2))
plt.plot(grids_pinn, density_pinn, label='PINN', linestyle='-', color='green', alpha=1, linewidth=2)
plt.plot(grids_deep, density_deep, label='PI-DeepONet', linestyle='-', color='red', alpha=1, linewidth=2)
plt.yscale("log")
plt.xscale("log")
plt.ylabel('Hessian Eigenvalue Density', fontsize=font)
plt.xlabel('Eigenvlaue', fontsize=font)
plt.xticks(fontsize=font)
plt.yticks(fontsize=font)
if fb == 10:
    plt.legend(fontsize=font)
plt.title(r"$F_B = {}$".format(fb), fontsize=font)
plt.tight_layout()

plt.savefig(
    f'/home/xinmeng/pinn_bow_mass/trained_model/fa25/export_figure/hessian_eigen_pinn_deeponet_fb{fb}.pdf', dpi=300, format="pdf", bbox_inches="tight")
plt.show()
print('')
# ax[0].plot(t_p_fd, p_fd , label='FDM', linestyle=':', color='black', alpha=1 , linewidth=1)
# ax[0].plot(t_pinn, p_pinn, label='PINN', linestyle='--', color='green', alpha=0.8, linewidth=1)
# ax[0].plot(t_deep, p_deep, label='PI-DeepONet', linestyle='--', color='red', alpha=0.5, linewidth=1)
# # ax[0].set_xlabel('Time [s]')
# ax[0].set_ylabel(r"$p \, [\mathrm{m}]$", fontsize=font)
# ax[0].set_xlim([0, 0.4])
# # ax[0].set_ylim([0.1, 0.4])
# # ax[0].legend(loc = 'best', fontsize=font)
# ax[0].legend(fontsize=font)
# # ax[0].legend(loc='upper center', bbox_to_anchor=(0., 1.26, 1., .102), fontsize=15)
# ax[0].set_title(r"$F_B = {}$".format(fb), fontsize=font)
# ax[0].tick_params(axis='both', which='major', labelsize=font)
#
# # ax[1].plot(t_train.detach().numpy(), q_exact, label='Exact Solution')
# ax[1].plot(t_q_fd, q_fd , label='FDM', linestyle=':', color='black', alpha=1, linewidth=1)
# ax[1].plot(t_pinn, q_pinn, label='PINN', linestyle='--', color='green', alpha=0.8, linewidth=1)
# ax[1].plot(t_deep, q_deep, label='PI-DeepONet', linestyle='--', color='red', alpha=0.5, linewidth=1)
# ax[1].set_xlabel('t [s]', fontsize=font)
# ax[1].set_ylabel(r"$q \, [\mathrm{m}/\mathrm{s}]$", fontsize=font)
# ax[1].set_xlim([0, 0.4])
# ax[1].tick_params(axis='both', which='major', labelsize=font)
# ax[1].set_ylim([0.01, 1])
# ax[1].set_xlim([np.min(t_train), np.max(t_train)])
# ax[1].legend()
# tikzplotlib.save(f"/home/xinmeng/pinn_bow_mass/trained_model/fa25/export_figure/fb{fb}.tex", axis_width="\\textwidth")
# ax[1].set_title(f'q: epoch {epoch}')
# plt.show()
# fig.savefig(f"/home/xinmeng/pinn_bow_mass/trained_model/fa25/export_figure/fb{fb}.pdf", dpi=300, format="pdf", bbox_inches="tight")
