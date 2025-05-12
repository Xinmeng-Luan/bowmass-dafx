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

def load_fd_result(path):
    p_q = loadmat(path)

    # t_p_fd = p_q['t_p'].squeeze()
    t_fd = p_q['t_q'].squeeze()
    p_fd = p_q['p'].squeeze()
    q_fd = p_q['q'].squeeze()

    new_indices = np.linspace(0, len(p_fd) - 1, len(p_fd)+1)
    p_fd = np.interp(new_indices, np.arange(len(p_fd)), p_fd)


    return t_fd, p_fd, q_fd

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

def get_pde_loss(t, p, q):
    vb = 0.2  # [m/s]
    a = 100
    omega = 2 * np.pi * 100
    eta = p - vb
    phi = np.sqrt(2 * a) * np.exp(-a * (eta ** 2) + 0.5) * eta
    pt = np.gradient(p, t)
    qt = np.gradient(q, t)
    res1 = pt + omega * q + fb * phi
    res2 = qt - omega * p
    return np.abs(res1), np.abs(res2)

fb = 1000
fd_path = f'../data/p_q_fd_05_fb_{fb}.mat'
fd_audiorate_path = f'../data/p_q_fd_05_fb_{fb}_audio_rate.mat'

deeponet_path = f"/home/xinmeng/pinn_bow_mass/trained_model/fa25/export_figure/fb_{fb}_deeponet.pkl"

pinn_path_1 = f"/home/xinmeng/pinn_bow_mass/trained_model/fa25/export_figure/fb_{fb}_pinn_nn1.pkl"
pinn_path_2 = f"/home/xinmeng/pinn_bow_mass/trained_model/fa25/export_figure/fb_{fb}_pinn_nn2.pkl"
pinn_path_3 = f"/home/xinmeng/pinn_bow_mass/trained_model/fa25/export_figure/fb_{fb}_pinn_nn3.pkl"
pinn_path_4 = f"/home/xinmeng/pinn_bow_mass/trained_model/fa25/export_figure/fb_{fb}_pinn_nn4.pkl"
pinn_path_5 = f"/home/xinmeng/pinn_bow_mass/trained_model/fa25/export_figure/fb_{fb}_pinn_nn5.pkl"


t_fd, p_fd, q_fd = load_fd_result(fd_path)
t_fd_audio, p_fd_audio, q_fd_audio = load_fd_result(fd_audiorate_path)
# t_pinn, p_pinn, q_pinn = merge_pinn_result(pinn_path_1, pinn_path_2, pinn_path_3)
t_pinn, p_pinn, q_pinn = merge_pinn_result(pinn_path_1, pinn_path_2, pinn_path_3, pinn_path_4, pinn_path_5)
t_deep, p_deep, q_deep = load_nn_result(deeponet_path)

res1_fd, res2_fd = get_pde_loss(t_fd, p_fd, q_fd)
res1_fd_audio, res2_fd_audio = get_pde_loss(t_fd_audio, p_fd_audio, q_fd_audio)
res1_pinn, res2_pinn = get_pde_loss(t_pinn, p_pinn, q_pinn)
res1_deep, res2_deep = get_pde_loss(t_deep, p_deep, q_deep)

font = 25
fig, ax = plt.subplots(2, 1, figsize=(5.5*2, 6*2))
# if fb != 1000:
ax[0].plot(t_deep, res1_deep, label='PI-DeepONet', linestyle='-', color='red', alpha=0.6, linewidth=1)
ax[0].plot(t_pinn, res1_pinn, label='PINN', linestyle='-', color='green', alpha=0.6, linewidth=1)
ax[0].plot(t_fd, res1_fd , label='FDM', linestyle='-', color='black', alpha=1 , linewidth=1)
ax[0].plot(t_fd_audio, res1_fd_audio , label='FDM-low', linestyle='-', color='blue', alpha=0.6 , linewidth=1)
ax[0].set_ylabel(r"ODE$_1$ Loss", fontsize=font)
if fb == 10:
    tmax = 0.3
elif  fb == 100:
    tmax = 0.09
elif fb == 1000:
    tmax = 0.0276
ax[0].set_xlim([0, tmax])
ax[0].set_title(r"$F_B = {}$".format(fb), fontsize=font)
ax[0].tick_params(axis='both', which='major', labelsize=font)
ax[0].set_yscale("log")
# if fb != 1000 :
ax[1].plot(t_deep, res2_deep, label='PI-DeepONet', linestyle='-', color='red', alpha=0.6, linewidth=1)
ax[1].plot(t_pinn, res2_pinn, label='PINN', linestyle='-', color='green', alpha=0.6, linewidth=1)
ax[1].plot(t_fd, res2_fd , label='FDM', linestyle='-', color='black', alpha=1, linewidth=1)
ax[1].plot(t_fd_audio, res2_fd_audio , label='FDM-low', linestyle='-', color='blue', alpha=0.6 , linewidth=1)
ax[1].set_xlabel('t [s]', fontsize=font)
ax[1].set_ylabel(r"ODE$_2$ Loss", fontsize=font)
ax[1].set_xlim([0, tmax])
ax[1].tick_params(axis='both', which='major', labelsize=font)
ax[1].set_yscale("log")
if fb == 1000:
    ax[1].legend(loc='best', fontsize=20)

plt.show()
fig.savefig(f"/home/xinmeng/pinn_bow_mass/trained_model/fa25/export_figure/res_fb{fb}.pdf", dpi=300, format="pdf", bbox_inches="tight")
print('')