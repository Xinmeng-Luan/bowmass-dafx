import pickle
import os
import sys
from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
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

def merge_pinn_result(pinn_path_1, pinn_path_2, pinn_path_3, pinn_path_4=None, pinn_path_5=None):
    t_list, p_list, q_list = [], [], []

    for path in [pinn_path_1, pinn_path_2, pinn_path_3, pinn_path_4, pinn_path_5]:
        if path is not None:
            t, p, q = load_nn_result(path)
            t_list.append(t)
            p_list.append(p)
            q_list.append(q)

    t = np.concatenate(t_list)
    p = np.concatenate(p_list)
    q = np.concatenate(q_list)

    return t, p, q

def nmse( p, q):
    # Compute MSE (Mean Squared Error)
    mse = np.mean((p - q) ** 2)

    # Compute the variance of p
    var_p = np.var(p)

    # Compute the NMSE
    nmse_value = mse / var_p
    return nmse_value

def ncc( p, q):
    # Compute the means of p and q
    p_mean = np.mean(p)
    q_mean = np.mean(q)

    # Compute the numerator of the NCC
    numerator = np.sum((p - p_mean) * (q - q_mean))

    # Compute the denominator of the NCC
    denominator = np.sqrt(np.sum((p - p_mean) ** 2) * np.sum((q - q_mean) ** 2))

    # Compute the NCC
    ncc_value = numerator / denominator
    return ncc_value

fb = 10

if fb == 1000:
    deeponet_path = f"../saved_data/saved_tpq/fb_{fb}_deeponet_hybrid.pkl"
else:
    deeponet_path = f"../saved_data/saved_tpq/fb_{fb}_deeponet.pkl"

pinn_path_1 = f"../saved_data/saved_tpq/fb_{fb}_pinn_nn1.pkl"
pinn_path_2 = f"../saved_data/saved_tpq/fb_{fb}_pinn_nn2.pkl"
pinn_path_3 = f"../saved_data/saved_tpq/fb_{fb}_pinn_nn3.pkl"
if fb==1000:
    pinn_path_4 = f"../saved_data/saved_tpq/fb_{fb}_pinn_nn4.pkl"
    pinn_path_5 = f"../saved_data/saved_tpq/fb_{fb}_pinn_nn5.pkl"
else:
    pinn_path_4 = None
    pinn_path_5 = None

t_p_fd, t_q_fd, p_fd, q_fd = load_fd_result(fb)
t_pinn, p_pinn, q_pinn = merge_pinn_result(pinn_path_1, pinn_path_2, pinn_path_3, pinn_path_4, pinn_path_5)
t_deep, p_deep, q_deep = load_nn_result(deeponet_path)


if fb ==10:
    mask_ind = 0.3
elif fb ==100:
    mask_ind = 0.09
elif fb == 1000:
    mask_ind = 0.0276



    # elif
fs = 44100


# masking
mask_deep = (t_deep >= 0) & (t_deep <= mask_ind)
t_deep_masked = t_deep[mask_deep]
p_deep_masked = p_deep[mask_deep]
q_deep_masked = q_deep[mask_deep]

mask = (t_p_fd >= 0) & (t_p_fd <= mask_ind)
t_fd_masked = t_p_fd[mask]
p_fd_masked = p_fd[mask]
q_fd_= np.interp(
    np.linspace(0, len(q_fd) - 1, len(q_fd) - 1),
    np.arange(len(q_fd)),
    q_fd
)
q_fd_masked = q_fd_[mask]

def resample( t,p,q):

    t_re = np.linspace(0, t[-1], int(fs * t[-1]), endpoint=False)
    # Resample using linear interpolation
    p_re = np.interp(t_re, t, p)
    q_re = np.interp(t_re, t, q)
    # interp_indices = np.linspace(0, mask_ind,  int(np.floor(fs * 0.3)))
    # t_re = np.interp(interp_indices,np.arange(len(t)),t)
    # p_re = np.interp(interp_indices, np.arange(len(p)), p)
    # q_re = np.interp(interp_indices, np.arange(len(q)), q)
    return t_re, p_re, q_re

t_fd_re, p_fd_re,q_fd_re= resample( t_fd_masked,p_fd_masked,q_fd_masked)
t_pinn_re, p_pinn_re,q_pinn_re= resample( t_pinn,p_pinn,q_pinn)
t_deep_re, p_deep_re,q_deep_re= resample(t_deep_masked,p_deep_masked,q_deep_masked)

omega = 2*np.pi*100

from scipy.io import savemat

savemat(f"../audio/f_{fb}/f_{fb}_fd.mat", {"u": q_fd_re/omega, "fs": fs})
savemat(f"../audio/f_{fb}/f_{fb}_pinn.mat", {"u": q_pinn_re/omega, "fs": fs})
savemat(f"../audio/f_{fb}/f_{fb}_deep.mat", {"u": q_deep_re/omega, "fs": fs})



print('')

# t_fd_resampled = np.interp(
#     interp_indices,
#     np.arange(len(t_fd_masked)),
#     t_fd_masked
# )
#
# p_fd_resampled = np.interp(
#     interp_indices,
#     np.arange(len(p_fd_masked)),
#     p_fd_masked
# )
#
# q_fd_resampled = np.interp(
#     interp_indices,
#     np.arange(len(q_fd_masked)),
#     q_fd_masked
# )
#
