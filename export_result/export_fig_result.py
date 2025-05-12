import pickle
import os
import sys
sys.path.append('/home/xinmeng/pinn_bow_mass')
from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt
# import tikzplotlib
## fb = 100

def load_nn_result(path):
    with open(path, "rb") as f:
        data = pickle.load(f)
    t = data[0]
    p = data[1]
    q = data[2]
    return t, p, q

def load_fd_result(fb):
    p_q = loadmat(f'../data/p_q_fd_05_fb_{fb}.mat')

    t_p_fd = p_q['t_p'].squeeze()
    t_q_fd = p_q['t_q'].squeeze()
    p_fd = p_q['p'].squeeze()
    q_fd = p_q['q'].squeeze()
    return t_p_fd, t_q_fd, p_fd, q_fd

def merge_pinn_result(pinn_path_1, pinn_path_2, pinn_path_3, pinn_path_4, pinn_path_5):
    t1, p1, q1 = load_nn_result(pinn_path_1)
    t2, p2, q2 = load_nn_result(pinn_path_2)
    t3, p3, q3 = load_nn_result(pinn_path_3)
    t4, p4, q4 = load_nn_result(pinn_path_4)
    t5, p5, q5 = load_nn_result(pinn_path_5)
    t = np.concatenate((t1, t2, t3, t4, t5))
    p = np.concatenate((p1, p2, p3, p4, p5))
    q = np.concatenate((q1, q2, q3, q4, q5))
    # t = np.concatenate((t1, t2, t3))
    # p = np.concatenate((p1, p2, p3))
    # q = np.concatenate((q1, q2, q3))
    return t, p, q

fb = 1000

deeponet_path = f"/home/xinmeng/pinn_bow_mass/trained_model/fa25/export_figure/fb_{fb}_deeponet_hybrid.pkl"

pinn_path_1 = f"/home/xinmeng/pinn_bow_mass/trained_model/fa25/export_figure/fb_{fb}_pinn_nn1.pkl"
pinn_path_2 = f"/home/xinmeng/pinn_bow_mass/trained_model/fa25/export_figure/fb_{fb}_pinn_nn2.pkl"
pinn_path_3 = f"/home/xinmeng/pinn_bow_mass/trained_model/fa25/export_figure/fb_{fb}_pinn_nn3.pkl"
pinn_path_4 = f"/home/xinmeng/pinn_bow_mass/trained_model/fa25/export_figure/fb_{fb}_pinn_nn4.pkl"
pinn_path_5 = f"/home/xinmeng/pinn_bow_mass/trained_model/fa25/export_figure/fb_{fb}_pinn_nn5.pkl"


t_p_fd, t_q_fd, p_fd, q_fd = load_fd_result(fb)
t_pinn, p_pinn, q_pinn = merge_pinn_result(pinn_path_1, pinn_path_2, pinn_path_3, pinn_path_4, pinn_path_5)
# t_pinn, p_pinn, q_pinn = merge_pinn_result(pinn_path_1, pinn_path_2, pinn_path_3)#, pinn_path_4, pinn_path_5)
t_deep, p_deep, q_deep = load_nn_result(deeponet_path)

font = 25
if fb == 10:
    xmax =0.4
elif fb ==100:
    xmax = 0.2
elif fb ==1000:
    xmax = 0.1
fig, ax = plt.subplots(2, 1, figsize=(8.5*2, 4*2)) #figsize=(5.5*2, 6*2)

ax[0].plot(t_p_fd, p_fd , label='FDM', linestyle=':', color='black', alpha=1 , linewidth=1)
ax[0].plot(t_pinn, p_pinn, label='PINN', linestyle='--', color='green', alpha=0.8, linewidth=1)
ax[0].plot(t_deep, p_deep, label='hybrid-DeepONet', linestyle='--', color='red', alpha=0.5, linewidth=1)
# ax[0].set_xlabel('Time [s]')
ax[0].set_ylabel(r"$p \, [\mathrm{m s}^{-1}]$", fontsize=font)
ax[0].set_xlim([0, xmax])
# ax[0].set_ylim([0.1, 0.4])
# ax[0].legend(loc = 'best', fontsize=font)
# if fb == 10:
#     ax[0].legend(fontsize=font)
ax[0].legend(loc='upper center', bbox_to_anchor=(0.35, 1.26, 1., .102), fontsize=font-2)
ax[0].set_title(r"$F_B = {}$".format(fb), fontsize=font)
ax[0].tick_params(axis='both', which='major', labelsize=font)

# ax[1].plot(t_train.detach().numpy(), q_exact, label='Exact Solution')
ax[1].plot(t_q_fd, q_fd , label='FDM', linestyle=':', color='black', alpha=1, linewidth=1)
ax[1].plot(t_pinn, q_pinn, label='PINN', linestyle='--', color='green', alpha=0.8, linewidth=1)
ax[1].plot(t_deep, q_deep, label='PI-DeepONet', linestyle='--', color='red', alpha=0.5, linewidth=1)
ax[1].set_xlabel('t [s]', fontsize=font)
ax[1].set_ylabel(r"$q \, [\mathrm{m}]$", fontsize=font)


ax[1].set_xlim([0, xmax])
ax[1].tick_params(axis='both', which='major', labelsize=font)
# ax[1].set_ylim([0.01, 1])
# ax[1].set_xlim([np.min(t_train), np.max(t_train)])
# ax[1].legend()
# tikzplotlib.save(f"/home/xinmeng/pinn_bow_mass/trained_model/fa25/export_figure/fb{fb}.tex", axis_width="\\textwidth")
# ax[1].set_title(f'q: epoch {epoch}')

fig.savefig(f"/home/xinmeng/pinn_bow_mass/trained_model/fa25/export_figure/fb{fb}_hybrid.pdf", dpi=300, format="pdf", bbox_inches="tight")
plt.show()
print('')