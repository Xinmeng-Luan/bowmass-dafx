'''
DAFx 2025: Physics-Informed Neural Network for Nonlinear Bow-String Friction
- Main Script for Training and Testing PI-DeepONet
- Author: Xinmeng Luan
- Contact: xinmeng.luan@mail.mcgill.ca
- Date: 11/05/2025
'''
import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from itertools import cycle
import utils.para_bow as para
from datetime import datetime
from scipy.io import loadmat
import time
from tqdm import tqdm
from utils.soap import SOAP
import utils.nn_deeponet_fourier as NN
import pickle
import matplotlib
matplotlib.use('module://backend_interagg')
from pyhessian import hessian # Hessian computation
from export_result.density_plot import get_esd_plot

os.environ['CUDA_ALLOW_GROWTH'] = 'True'
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

# set seed of random numpy
np.random.seed(para.seed)
torch.manual_seed(para.seed)
torch.cuda.manual_seed(para.seed)
torch.cuda.manual_seed_all(para.seed)
# When running on the CuDNN backend, two further options must be set
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
# Set a fixed value for the hash seed
os.environ["PYTHONHASHSEED"] = str(para.seed)
torch.use_deterministic_algorithms(True)
print(f"Random seed set as {para.seed}")

# Check if GPUs are available
print("Check GPUs----")
try:
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        print(f"Number of available GPUs: {device_count}")
        device = torch.device('cuda:0')

    else:
        raise Exception("No GPUs available.")
except Exception as e:
    print(f"GPU Error: {str(e)}")
    sys.exit()

now = datetime.now()
print(now.strftime("%Y-%m-%d %H:%M:%S"))



# Train the PINN
class Trainer:
    def __init__(self, isvisual, isfourier, fb_value, pq_max, time_length, fourier_sigma, layer_size_branch, layer_size_trunk, optimizer_type):
        self.starter_learning_rate = 1e-3
        self.decay_rate = 0.9
        self.decay_steps = 3000 #5000
        self.n_epochs = 10001
        self.f = 100
        self.Fb = fb_value
        self.time_length = time_length
        self.omega = 2 * np.pi * self.f
        self.N_period = 10 # N periods
        self.T = 1/ self.f * self.N_period
        self.N_sample = 1000
        self.batch_size_res = 200000
        self.batch_size_ic = 2000
        if self.Fb == 1000:
            self.batch_size = 20000
        else:
            self.batch_size = 50000 #ori: 10000
        self.N_batch = 10
        self.tot_losses = []
        self.res_losses = []
        self.ic1_losses = []
        self.ic2_losses = []
        self.res1_losses = []
        self.res2_losses = []
        self.w_res1s = []
        self.w_res2s = []
        self.w_ic1s = []
        self.w_ic2s = []
        self.w_data1s = []
        self.w_data2s = []
        self.isvisual = isvisual
        self.isfourier = isfourier
        self.Q = 1000
        self.N = 10000 #soap: 20000

        self.pq_max = pq_max
        self.fourier_sigma = fourier_sigma
        self.layer_size_branch = layer_size_branch
        self.layer_size_trunk = layer_size_trunk
        self.optimizer_type = optimizer_type



    def pre_proc(self):
        # if self.isfourier:


        # else:
        #     import main.network.nn_deeponet as NN
        #     layer_size_branch = [2, 100, 100, 100, 100, 100, 100, 100, 200]
        #     layer_size_trunk = [1, 100, 100, 100, 100, 100, 100, 100, 200]
        self.model = NN.DeepOnet(self.layer_size_branch, self.layer_size_trunk, self.pq_max, self.time_length, self.fourier_sigma).to(device)
        N_total_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Total trainable parameters: {N_total_params}")

        lambda_lr = lambda step: self.decay_rate ** (step / self.decay_steps)
        if self.optimizer_type == 'adam':
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.starter_learning_rate)
            self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lambda_lr)
        elif self.optimizer_type == 'soap':
            self.optimizer = SOAP(self.model.parameters(), lr=3e-3, betas=(.95, .95), weight_decay=.01, precondition_frequency=10)
            self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lambda_lr)

        self.start_time = time.time()
        if self.isvisual:
            self.load_fd_result()
            self.p_vis = []
            self.q_vis = []

    # Define the physics-informed loss (PDE residual)
    def res_loss(self, t, pq0):

        p, q = self.model(t, pq0)
        vb = 0.2  # [m/s]
        # Fb = 100  # 100, 4000
        a = 100
        eta = p - vb
        phi = torch.sqrt(2 * torch.tensor(a)) * torch.exp(-a * (eta ** 2) + 0.5) * eta

        p_t = (torch.autograd.grad(p, t, grad_outputs=torch.ones_like(p), create_graph=True)[0])
        q_t = (torch.autograd.grad(q, t, grad_outputs=torch.ones_like(q), create_graph=True)[0])

        # pde loss
        pde_residual1 = p_t + self.omega * q + self.Fb * phi
        pde_residual2 = q_t - self.omega * p

        pde1_loss = torch.mean((pde_residual1 - 0) ** 2)
        pde2_loss = torch.mean((pde_residual2 - 0) ** 2)
        pde_loss = pde1_loss + pde2_loss

        return pde_loss

    def res_loss_sep(self, t, pq0):

        p, q = self.model(t, pq0)
        vb = 0.2  # [m/s]
        # Fb = 100  # 100, 4000
        a = 100
        eta = p - vb
        phi = torch.sqrt(2 * torch.tensor(a)) * torch.exp(-a * (eta ** 2) + 0.5) * eta

        p_t = (torch.autograd.grad(p, t, grad_outputs=torch.ones_like(p), create_graph=True)[0])
        q_t = (torch.autograd.grad(q, t, grad_outputs=torch.ones_like(q), create_graph=True)[0])

        # pde loss
        pde_residual1 = p_t + self.omega * q + self.Fb * phi
        pde_residual2 = q_t - self.omega * p

        pde1_loss = torch.mean((pde_residual1 - 0) ** 2)
        pde2_loss = torch.mean((pde_residual2 - 0) ** 2)


        return pde1_loss, pde2_loss

    def data_loss(self, t, pq0, p_gt, q_gt):

        p, q = self.model(t, pq0)
        # vb = 0.2  # [m/s]
        # # Fb = 100  # 100, 4000
        # a = 100
        # eta = p - vb
        # phi = torch.sqrt(2 * torch.tensor(a)) * torch.exp(-a * (eta ** 2) + 0.5) * eta
        #
        # p_t = (torch.autograd.grad(p, t, grad_outputs=torch.ones_like(p), create_graph=True)[0])
        # q_t = (torch.autograd.grad(q, t, grad_outputs=torch.ones_like(q), create_graph=True)[0])

        # # pde loss
        # pde_residual1 = p_t + self.omega * q + self.Fb * phi
        # pde_residual2 = q_t - self.omega * p
        #
        # pde1_loss = torch.mean((pde_residual1 - 0) ** 2)
        # pde2_loss = torch.mean((pde_residual2 - 0) ** 2)
        data_loss1 = torch.mean((p.squeeze() - p_gt.squeeze()) ** 2)
        data_loss2 = torch.mean((q.squeeze() - q_gt.squeeze()) ** 2)


        return data_loss1, data_loss2


    # Initial condition loss
    def ic_loss(self, t, pq0, pq0_gt):
        p, q = self.model(t, pq0)


        ic_loss1 = torch.mean((p.squeeze() - pq0_gt[:,0]) ** 2)
        ic_loss2 = torch.mean((q.squeeze() - pq0_gt[:,1]) ** 2)

        return ic_loss1, ic_loss2


    def save_model(self, model, optimizer, epoch, tot_losses, res_losses, ic1_losses, ic2_losses, path):
        end_time = time.time()
        elapsed_time = end_time - self.start_time

        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch': epoch,
            'tot_losses': tot_losses,
            'res_losses': res_losses,
            'ic1_losses': ic1_losses,
            'ic2_losses': ic2_losses,
            'elapsed_time': elapsed_time

        }, path)

    def save_model_anneal(self, model, optimizer, epoch, tot_losses, res1_losses, res2_losses,
                          ic1_losses, ic2_losses,
                          w_res1s, w_res2s, w_ic1s, w_ic2s,
                          path):
        end_time = time.time()
        elapsed_time = end_time - self.start_time

        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch': epoch,
            'tot_losses': tot_losses,
            'res1_losses': res1_losses,
            'res2_losses': res2_losses,
            'ic1_losses': ic1_losses,
            'ic2_losses': ic2_losses,
            'res1_ws': w_res1s,
            'res2_wss': w_res2s,
            'ic1_ws': w_ic1s,
            'ic2_ws': w_ic2s,
            'elapsed_time': elapsed_time

        }, path)

    def save_model_anneal_data(self, model, optimizer, epoch, tot_losses, res1_losses, res2_losses,
                          ic1_losses, ic2_losses,
                          w_res1s, w_res2s, w_ic1s, w_ic2s, w_data1s, w_data2s,
                          path):
        end_time = time.time()
        elapsed_time = end_time - self.start_time

        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch': epoch,
            'tot_losses': tot_losses,
            'res1_losses': res1_losses,
            'res2_losses': res2_losses,
            'ic1_losses': ic1_losses,
            'ic2_losses': ic2_losses,
            'res1_ws': w_res1s,
            'res2_wss': w_res2s,
            'ic1_ws': w_ic1s,
            'ic2_ws': w_ic2s,
            'data1_ws': w_data1s,
            'data2_ws': w_data2s,
            'elapsed_time': elapsed_time

        }, path)

    def get_dataset_from_fd(self):
        ds = loadmat('../data/input_ds_discrete_fix_fb_1000_one.mat')
        p0 = torch.from_numpy(ds['p']).type(torch.float32)
        q0 = torch.from_numpy(ds['q']).type(torch.float32)
        t_res = torch.zeros_like(p0) * self.time_length
        # t_res = torch.rand_like(p0) * self.time_length
        t_ic = torch.zeros_like(p0) * self.time_length
        pq0 = torch.cat((p0, q0), dim=1)

        ic_dataset = TensorDataset(t_ic, pq0, pq0)
        res_dataset = TensorDataset(t_res, pq0)
        # res_loader = DataLoader(res_dataset, self.batch_size, shuffle=True)
        # ic_loader = DataLoader(ic_dataset, self.batch_size, shuffle=True)
        res_loader = DataLoader(res_dataset, int(441000/2), shuffle=True)
        ic_loader = DataLoader(ic_dataset, int(441000/2), shuffle=True)
        return res_loader, ic_loader

    def get_one_dataset_train(self):
        # Initial condition (IC) sample
        # if self.Fb == 1000:
        #     pq0_ic = torch.zeros(1, 2)
        # else:
        pq0_ic = torch.rand(1, 2) * self.pq_max*2 -self.pq_max
        t_ic = torch.tensor([0.0])            # [0, 1]
        pq0_ic_gt = pq0_ic                     # Ground truth same as IC sample

        # Residual (Res) samples

        pq0_res = pq0_ic.repeat(self.Q, 1)    # Repeat IC sample Q times
        t_res = torch.rand(self.Q)  * self.time_length         # Random time samples [0, 1]

        return pq0_ic, t_ic, pq0_ic_gt, pq0_res, t_res

    def get_dataset_train(self):
        pq0_ic_list = []
        t_ic_list = []
        pq0_ic_gt_list = []
        pq0_res_list = []
        t_res_list = []

        for _ in range(self.N):
            pq0_ic, t_ic, pq0_ic_gt, pq0_res, t_res = self.get_one_dataset_train()
            pq0_ic_list.append(pq0_ic)
            t_ic_list.append(t_ic)
            pq0_ic_gt_list.append(pq0_ic_gt)
            pq0_res_list.append(pq0_res)
            t_res_list.append(t_res)

        # Stack data into tensors
        pq0_ic = torch.cat(pq0_ic_list, dim=0)       # Shape: [N, 2]
        t_ic = torch.cat(t_ic_list, dim=0).unsqueeze(1)           # Shape: [N]
        pq0_ic_gt = torch.cat(pq0_ic_gt_list, dim=0) # Shape: [N, 2]
        pq0_res = torch.cat(pq0_res_list, dim=0)     # Shape: [N*Q, 2]
        t_res = torch.cat(t_res_list, dim=0).unsqueeze(1)       # Shape: [N*Q]

        # plt.scatter(pq0_ic[:, 0].detach().cpu().numpy(), pq0_ic[:, 1].detach().cpu().numpy(), s=0.1), plt.show()
        ic_dataset = TensorDataset(t_ic, pq0_ic, pq0_ic_gt)
        res_dataset = TensorDataset(t_res, pq0_res)
        res_loader = DataLoader(res_dataset, self.batch_size, shuffle=True)
        ic_loader = DataLoader(ic_dataset, self.batch_size, shuffle=True)

        return res_loader, ic_loader

    def get_dataset(self):
        res_loader, ic_loader = self.get_dataset_train()

        return res_loader, ic_loader

    def get_data_driven_ds(self):
        ds = loadmat('../data/data_driven_fd_fb_1000.mat')
        p_fd = torch.from_numpy(ds['p']).type(torch.float32).squeeze()
        q_fd = torch.from_numpy(ds['q']).type(torch.float32).squeeze()
        downsample_factor = 100
        p_fd = p_fd[::downsample_factor]
        q_fd = q_fd[::downsample_factor]
        # t_fd = torch.from_numpy(ds['t']).type(torch.float32)
        fs_ori = 44100*100
        fs = fs_ori/downsample_factor
        num_per_window = int(self.time_length * fs)

        t_ref = torch.linspace(0,self.time_length, (num_per_window))
        p_gt, q_gt, t = [], [], []
        p0, q0 = [], []

        for ii in range(p_fd.shape[0] - (num_per_window)):
            p_gt.append(p_fd[ii:ii + num_per_window])
            q_gt.append(q_fd[ii:ii + num_per_window])
            t.append(t_ref)
            # p0.append(p_fd[ii].repeat(num_per_window))
            # q0.append(q_fd[ii].repeat(num_per_window))
            p0.append(p_fd[ii].repeat(num_per_window))
            q0.append(q_fd[ii].repeat(num_per_window))

        # Convert to tensors
        p_gt = torch.cat(p_gt,dim=0).unsqueeze(1)
        q_gt = torch.cat(q_gt,dim=0).unsqueeze(1)
        p0 = torch.cat(p0,dim=0).unsqueeze(1)
        q0 = torch.cat(q0,dim=0).unsqueeze(1)
        t = torch.cat(t,dim=0).unsqueeze(1)
        pq0 =  torch.cat((p0, q0), dim=1)

        res_dataset = TensorDataset(t, pq0, p_gt, q_gt)
        res_loader = DataLoader(res_dataset, self.batch_size, shuffle=True)

        return res_loader#, ic_loader

    def test_visual(self, epoch):
        # self.pre_proc()
        t = self.t_test.to(device).requires_grad_()
        t = t.unsqueeze(1)
        p0_value = self.omega
        q0_value = 0

        for time_seg in range(int(0.1/self.time_length)):
            self.optimizer.zero_grad()

            p0 = p0_value * torch.ones(self.batch_size_ic, 1).to(device).requires_grad_()
            q0 = q0_value * torch.ones(self.batch_size_ic, 1).to(device).requires_grad_()
            pq0 = torch.cat((p0, q0), dim=1)

            p, q = self.model(t, pq0)

            p0_value = p.squeeze()[-1]
            q0_value = q.squeeze()[-1]

            if time_seg != 0:
                p = p[1:]
                q = q[1:]
            self.p_vis.extend(p.squeeze().detach().cpu().tolist())
            self.q_vis.extend(q.squeeze().detach().cpu().tolist())

        self.visualize(np.array(self.p_vis), np.array(self.q_vis), epoch)


        return self.model
    def train(self):
        self.pre_proc()
        # if self.Fb == 1000:
        #     train_res_loader, train_ic_loader = self.get_dataset_from_fd()
        #     print('')
        #     # train_res_loader, train_ic_loader = self.get_dataset()
        # else:
        train_res_loader, train_ic_loader = self.get_dataset()
        ic_loader_cycle = cycle(train_ic_loader)
        itr = 0
        for epoch in tqdm(range(self.n_epochs), desc="Training Progress (Epochs)"):
            total_loss, total_loss_res, total_loss_ic1, total_loss_ic2 = 0, 0, 0, 0

            for (t_res, pq0_res) in tqdm(train_res_loader, desc="Batch Progress", leave=False):
                (t_ic, pq0_ic, pq0_ic_gt) = next(ic_loader_cycle)  # cycle ic dataset

                t_res, pq0_res, t_ic, pq0_ic, pq0_ic_gt = (
                    t_res.to(device).requires_grad_(),
                    pq0_res.to(device).requires_grad_(),
                    # pq0_res_gt.to(device),
                    t_ic.to(device).requires_grad_(),
                    pq0_ic.to(device).requires_grad_(),
                    pq0_ic_gt.to(device).requires_grad_()
                )


                self.optimizer.zero_grad()

                # Compute residual and initial condition losses
                loss_res = self.res_loss(t_res, pq0_res)
                loss_ic1, loss_ic2 = self.ic_loss(t_ic, pq0_ic, pq0_ic_gt)
                # loss_res = loss_res / (0.03 ** 2)
                # loss_ic1 = loss_ic1 / 1e-8
                # loss_ic2 = loss_ic2 / 1e-8

                loss = loss_res * 1e1  + loss_ic1 * 1e6 + loss_ic2 * 1e6

                loss.backward()
                self.optimizer.step()
                if self.optimizer_type == 'adam':
                    self.scheduler.step()

                batch_loss = loss.detach().cpu().item()
                batch_loss_res = loss_res.detach().cpu().item()
                batch_loss_ic1 = loss_ic1.detach().cpu().item()
                batch_loss_ic2 = loss_ic2.detach().cpu().item()


                self.tot_losses.append(batch_loss)
                self.res_losses.append(batch_loss_res)
                self.ic1_losses.append(batch_loss_ic1)
                self.ic2_losses.append(batch_loss_ic2)
                itr =  itr+1


            if epoch % 1 == 0:
                print(f'Epoch {itr}, loss_res: {batch_loss_res:.4e}, '
                      f'loss_ic1: {batch_loss_ic1:.4e}, loss_ic2: {batch_loss_ic2:.4e}')
                if self.isvisual:
                    self.test_visual(epoch)

            if (epoch + 1) % 1 == 0:
                save_path = (f'/home/xinmeng/pinn_bow_mass/trained_model/fa25/deeponet/'
                             f'dafx_no_anneal_{self.optimizer_type}_oneds_deeponet_bowmass_fb{self.Fb}_timelength_{self.time_length}_isfourier_{self.isfourier}_sigma_{self.fourier_sigma}_itr{itr}.pth')

                # save_path = (f'/home/xinmeng/pinn_bow_mass/trained_model/fa25/'
                #              f'ic0_oneds_deeponet_bowmass_{self.optimizer_type}_fb{self.Fb}_timelength_{self.time_length}_isfourier_{self.isfourier}_{epoch + 1}.pth')
                self.save_model(self.model, self.optimizer, epoch+1, self.tot_losses,
                                self.res_losses, self.ic1_losses, self.ic2_losses, save_path)
                print(f'Model saved at epoch: {epoch + 1}.')

        return self.model

    # def loss_annealing(self, loss_res1, loss_res2, loss_ic1, loss_ic2):
    #     # Compute gradient norms for balancing
    #     loss_res1_norm = torch.autograd.grad(loss_res1, self.model.parameters(), retain_graph=True)
    #     loss_res2_norm = torch.autograd.grad(loss_res2, self.model.parameters(), retain_graph=True)
    #     loss_ic1_norm = torch.autograd.grad(loss_ic1, self.model.parameters(), retain_graph=True)
    #     loss_ic2_norm = torch.autograd.grad(loss_ic2, self.model.parameters(), retain_graph=True)
    #
    #     loss_res1_norm = sum(p.norm(2) for p in loss_res1_norm if p is not None)
    #     loss_res2_norm = sum(p.norm(2) for p in loss_res2_norm if p is not None)
    #     loss_ic1_norm = sum(p.norm(2) for p in loss_ic1_norm if p is not None)
    #     loss_ic2_norm = sum(p.norm(2) for p in loss_ic2_norm if p is not None)
    #
    #     # Compute dynamic weights
    #     total_norm = loss_res1_norm + loss_res2_norm + loss_ic1_norm + loss_ic2_norm
    #     w_res1 = total_norm / (loss_res1_norm + 1e-20)
    #     w_res2 = total_norm / (loss_res2_norm + 1e-20)
    #     w_ic1 = total_norm / (loss_ic1_norm + 1e-20)
    #     w_ic2 = total_norm / (loss_ic2_norm + 1e-20)
    #
    #     return w_res1, w_res2, w_ic1, w_ic2

    def loss_annealing(self, *losses):
        # Compute gradient norms for each loss
        loss_norms = []
        for loss in losses:
            grad_norm = torch.autograd.grad(loss, self.model.parameters(), retain_graph=True)
            grad_norm = sum(p.norm(2) for p in grad_norm if p is not None)  # L2 norm
            loss_norms.append(grad_norm)

        total_norm = sum(loss_norms)
        weights = [total_norm / (norm + 1e-20) for norm in loss_norms]

        return weights
    def train_lr_anneal(self):
        self.pre_proc()

        train_res_loader, train_ic_loader = self.get_dataset()

        itr = 0
        for epoch in tqdm(range(self.n_epochs), desc="Training Progress (Epochs)"):
            ic_loader_cycle = cycle(train_ic_loader)
            for (t_res, pq0_res) in tqdm(train_res_loader, desc="Batch Progress", leave=False):

                (t_ic, pq0_ic, pq0_ic_gt) = next(ic_loader_cycle)  # cycle ic dataset

                t_res, pq0_res, t_ic, pq0_ic, pq0_ic_gt = (
                    t_res.to(device).requires_grad_(),
                    pq0_res.to(device).requires_grad_(),
                    # pq0_res_gt.to(device),
                    t_ic.to(device).requires_grad_(),
                    pq0_ic.to(device).requires_grad_(),
                    pq0_ic_gt.to(device).requires_grad_()
                )


                self.optimizer.zero_grad()

                # Compute residual and initial condition losses
                loss_res1, loss_res2 = self.res_loss_sep(t_res, pq0_res)
                loss_ic1, loss_ic2 = self.ic_loss(t_ic, pq0_ic, pq0_ic_gt)

                if itr % 100 == 0:
                    w_res1, w_res2, w_ic1, w_ic2 = self.loss_annealing(loss_res1, loss_res2, loss_ic1, loss_ic2)

                # Compute final loss
                loss = w_res1 * loss_res1 + w_res2 * loss_res2 + w_ic1 * loss_ic1 + w_ic2 * loss_ic2

                loss.backward()
                self.optimizer.step()
                self.scheduler.step()

                batch_loss = loss.detach().cpu().item()
                batch_loss_res1 = loss_res1.detach().cpu().item()
                batch_loss_res2 = loss_res2.detach().cpu().item()
                batch_loss_ic1 = loss_ic1.detach().cpu().item()
                batch_loss_ic2 = loss_ic2.detach().cpu().item()
                batch_w_res1 = w_res1.detach().cpu().item()
                batch_w_res2 = w_res2.detach().cpu().item()
                batch_w_ic1 = w_ic1.detach().cpu().item()
                batch_w_ic2 = w_ic2.detach().cpu().item()


                self.tot_losses.append(batch_loss)
                self.res1_losses.append(batch_loss_res1)
                self.res2_losses.append(batch_loss_res2)
                self.ic1_losses.append(batch_loss_ic1)
                self.ic2_losses.append(batch_loss_ic2)
                self.w_res1s.append(batch_w_res1)
                self.w_res2s.append(batch_w_res2)
                self.w_ic1s.append(batch_w_ic1)
                self.w_ic2s.append(batch_w_ic2)

                itr = itr + 1


            if epoch % 1 == 0:
                print(f'Epoch {epoch}, loss_res1: {batch_loss_res1:.4e}, loss_res2: {batch_loss_res2:.4e}, '
                      f'loss_ic1: {batch_loss_ic1:.4e}, loss_ic2: {batch_loss_ic2:.4e}, '
                      f'w_res1: {batch_w_res1:.4e}, w_res2: {batch_w_res2:.4e}, '
                      f'w_ic1: {batch_w_ic1:.4e}, w_ic2: {batch_w_ic2:.4e}')
                if self.isvisual:
                    self.test_visual(epoch)

            if (epoch + 1) % 1 == 0:
                save_path = (f'/home/xinmeng/pinn_bow_mass/trained_model/fa25/deeponet/'
                             f'dafx_lr_anneal_{self.optimizer_type}_oneds_deeponet_bowmass_fb{self.Fb}_timelength_{self.time_length}_isfourier_{self.isfourier}_sigma_{self.fourier_sigma}_itr{itr}.pth')
                self.save_model_anneal(self.model, self.optimizer, epoch+1, self.tot_losses,
                                self.res1_losses, self.res2_losses, self.ic1_losses, self.ic2_losses,
                                       self.w_res1s, self.w_res2s, self.w_ic1s, self.w_ic2s,
                                       save_path)
                print(f'Model saved at epoch: {epoch + 1}.')

        return self.model

    def train_lr_anneal_hybrid(self):
        self.pre_proc()

        train_res_loader, train_ic_loader = self.get_dataset()
        data_driven_loader = self.get_data_driven_ds()

        itr = 0
        for epoch in tqdm(range(self.n_epochs), desc="Training Progress (Epochs)"):
            ic_loader_cycle = cycle(train_ic_loader)
            data_driven_loader_cycle = cycle(data_driven_loader)
            for (t_res, pq0_res) in tqdm(train_res_loader, desc="Batch Progress", leave=False):

                (t_ic, pq0_ic, pq0_ic_gt) = next(ic_loader_cycle)  # cycle ic dataset
                (t_data, pq0_data, p_gt_data, q_gt_data) = next(data_driven_loader_cycle)

                (t_res, pq0_res, t_ic, pq0_ic, pq0_ic_gt,
                 t_data, pq0_data, p_gt_data, q_gt_data) = (
                    t_res.to(device).requires_grad_(),
                    pq0_res.to(device).requires_grad_(),
                    # pq0_res_gt.to(device),
                    t_ic.to(device).requires_grad_(),
                    pq0_ic.to(device).requires_grad_(),
                    pq0_ic_gt.to(device).requires_grad_(),
                    t_data.to(device).requires_grad_(),
                    pq0_data.to(device).requires_grad_(),
                    p_gt_data.to(device).requires_grad_(),
                    q_gt_data.to(device).requires_grad_()
                )

                self.optimizer.zero_grad()

                # Compute residual and initial condition losses
                loss_res1, loss_res2 = self.res_loss_sep(t_res, pq0_res)
                loss_ic1, loss_ic2 = self.ic_loss(t_ic, pq0_ic, pq0_ic_gt)
                loss_data1, loss_data2 = self.data_loss(t_data, pq0_data, p_gt_data, q_gt_data)

                if itr % 100 == 0:
                    w_res1, w_res2, w_ic1, w_ic2, w_data1, w_data2 =\
                        self.loss_annealing(loss_res1, loss_res2, loss_ic1, loss_ic2, loss_data1, loss_data2)

                # Compute final loss
                loss = (w_res1 * loss_res1 + w_res2 * loss_res2 + w_ic1 * loss_ic1 + w_ic2 * loss_ic2 +
                        w_data1 * loss_data1 + w_data2 * loss_data2)

                loss.backward()
                self.optimizer.step()
                self.scheduler.step()

                batch_loss = loss.detach().cpu().item()
                batch_loss_res1 = loss_res1.detach().cpu().item()
                batch_loss_res2 = loss_res2.detach().cpu().item()
                batch_loss_ic1 = loss_ic1.detach().cpu().item()
                batch_loss_ic2 = loss_ic2.detach().cpu().item()
                batch_w_res1 = w_res1.detach().cpu().item()
                batch_w_res2 = w_res2.detach().cpu().item()
                batch_w_ic1 = w_ic1.detach().cpu().item()
                batch_w_ic2 = w_ic2.detach().cpu().item()
                batch_w_data1 = w_data1.detach().cpu().item()
                batch_w_data2 = w_data2.detach().cpu().item()

                self.tot_losses.append(batch_loss)
                self.res1_losses.append(batch_loss_res1)
                self.res2_losses.append(batch_loss_res2)
                self.ic1_losses.append(batch_loss_ic1)
                self.ic2_losses.append(batch_loss_ic2)
                self.w_res1s.append(batch_w_res1)
                self.w_res2s.append(batch_w_res2)
                self.w_ic1s.append(batch_w_ic1)
                self.w_ic2s.append(batch_w_ic2)
                self.w_data1s.append(batch_w_data1)
                self.w_data2s.append(batch_w_data2)

                itr = itr + 1


            if epoch % 1 == 0:
                print(f'Epoch {epoch}, loss_res1: {batch_loss_res1:.4e}, loss_res2: {batch_loss_res2:.4e}, '
                      f'loss_ic1: {batch_loss_ic1:.4e}, loss_ic2: {batch_loss_ic2:.4e}, '
                      f'w_res1: {batch_w_res1:.4e}, w_res2: {batch_w_res2:.4e}, '
                      f'w_ic1: {batch_w_ic1:.4e}, w_ic2: {batch_w_ic2:.4e},'
                      f'w_data1: {batch_w_data1:.4e}, w_data2: {batch_w_data2:.4e}')
                if self.isvisual:
                    self.test_visual(epoch)

            if (epoch + 1) % 1 == 0:
                save_path = (f'/home/xinmeng/pinn_bow_mass/trained_model/fa25/deeponet/'
                             f'dafx_hybrid_lr_anneal_{self.optimizer_type}_oneds_deeponet_bowmass_fb{self.Fb}_timelength_{self.time_length}_isfourier_{self.isfourier}_sigma_{self.fourier_sigma}_itr{itr}.pth')
                self.save_model_anneal_data(self.model, self.optimizer, epoch+1, self.tot_losses,
                                self.res1_losses, self.res2_losses, self.ic1_losses, self.ic2_losses,
                                       self.w_res1s, self.w_res2s, self.w_ic1s, self.w_ic2s, self.w_data1s, self.w_data2s,
                                       save_path)
                print(f'Model saved at epoch: {epoch + 1}.')

        return self.model

    def load_fd_result(self):
        p_q = loadmat(f'../data/p_q_fd_05_fb_{self.Fb}.mat')

        # p_q = loadmat('../data/p_q_fd_01.mat')
        self.p_fd = p_q['p'].squeeze()
        self.q_fd = p_q['q'].squeeze()
        self.t_p_fd = p_q['t_p'].squeeze()
        self.t_q_fd = p_q['t_q'].squeeze()

    def visualize(self,  p_out, q_out, epoch, tmax):

        fig, ax = plt.subplots(2, 1, figsize=(8, 6))
        t_train = np.linspace(0, tmax, num=len(p_out))
        # ax[0].plot(t_train.detach().numpy(), p_exact, label='Exact Solution')
        ax[0].plot(t_train, p_out, label='DeepOnet', linestyle=':')
        ax[0].plot(self.t_p_fd, self.p_fd, label='FD', linestyle='-')
        ax[0].set_xlabel('Time [s]')
        ax[0].set_ylabel('p (t)')
        ax[0].set_xlim([0, tmax])
        # ax[0].set_ylim([self.p_fd.min(), self.p_fd.max()])
        ax[0].legend()
        ax[0].set_title(f'p: epoch {epoch}')

        # ax[1].plot(t_train.detach().numpy(), q_exact, label='Exact Solution')
        ax[1].plot(t_train, q_out, label='DeepOnet', linestyle=':')
        ax[1].plot(self.t_q_fd, self.q_fd, label='FD', linestyle='-')
        ax[1].set_xlabel('Time [s]')
        ax[1].set_ylabel('q (t)')
        ax[1].set_xlim([0, tmax])
        # ax[1].set_ylim([self.q_fd.min(), self.q_fd.max()])
        ax[1].legend()
        ax[1].set_title(f'q: epoch {epoch}')

        # Adjust layout and show the figure
        # plt.tight_layout()
        # plt.savefig("/home/xinmeng/pinn_bow_mass/trained_model/fa25/test.png", dpi=300, bbox_inches="tight")
        plt.show()
        print('')

    def test(self, model_path):
        self.pre_proc()
        self.p_vis = []
        self.q_vis = []
        checkpoint = torch.load(model_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        # self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        epoch = checkpoint['epoch']
        tot_losses = checkpoint['tot_losses']
        res_losses = checkpoint['res_losses']
        ic1_losses = checkpoint['ic1_losses']
        ic2_losses = checkpoint['ic2_losses']

        tot_losses = [loss for loss in tot_losses]
        res_losses = [loss for loss in res_losses]
        ic1_losses = [loss for loss in ic1_losses]
        ic2_losses = [loss for loss in ic2_losses]

        # Plot loss
        plt.figure(figsize=(10, 6))

        plt.plot( tot_losses, label='Total Loss', color='blue')
        plt.plot( res_losses, label='Residual Loss', color='red')
        plt.plot( ic1_losses, label='IC1 Loss', color='green')
        plt.plot( ic2_losses, label='IC2 Loss', color='orange')

        plt.yscale('log')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Training Losses Over Epochs')
        plt.legend()
        plt.grid(True)
        plt.show()

        self.Q = 100
        t = torch.linspace(0, self.time_length, self.Q).to(device).requires_grad_()
        t = t.unsqueeze(1)
        # t = torch.rand(self.Q, 1).to(device).requires_grad_()
        p0_value = 0
        q0_value = 0
        tmax = 0.5
        for time_seg in range(int(tmax / self.time_length)):
            self.optimizer.zero_grad()

            p0 = p0_value * torch.ones(self.Q, 1).to(device).requires_grad_()
            q0 = q0_value * torch.ones(self.Q, 1).to(device).requires_grad_()
            pq0 = torch.cat((p0, q0), dim=1)

            p, q = self.model(t, pq0)

            p0_value = p.squeeze()[-1]
            q0_value = q.squeeze()[-1]

            if time_seg != 0:
                p = p[1:]
                q = q[1:]
            self.p_vis.extend(p.squeeze().detach().cpu().tolist())
            self.q_vis.extend(q.squeeze().detach().cpu().tolist())

        self.load_fd_result()
        self.visualize(np.array(self.p_vis), np.array(self.q_vis), epoch, tmax=tmax)

    def test_hessian(self, model_path):
        self.pre_proc()
        checkpoint = torch.load(model_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        # train_res_loader, train_ic_loader = self.get_dataset()

        res1_ws = checkpoint['res1_ws'][-1]
        res2_ws = checkpoint['res2_wss'][-1]
        ic1_ws = checkpoint['ic1_ws'][-1]
        ic2_ws = checkpoint['ic2_ws'][-1]
        #


        # w_res1, w_res2, w_ic1, w_ic2 = self.loss_annealing(loss_res1, loss_res2, loss_ic1, loss_ic2)

        t = torch.linspace(0, self.time_length, 5000).requires_grad_(True).to(
            device).unsqueeze(1)
        p0_value = 0
        q0_value = 0
        p0 = p0_value * torch.ones(5000, 1).to(device).requires_grad_()
        q0 = q0_value * torch.ones(5000, 1).to(device).requires_grad_()
        pq0 = torch.cat((p0, q0), dim=1)
        data = [t, pq0]
        self.optimizer.zero_grad()
        import hessian_compute_deeponet
        import density_plot
        hessian_comp = hessian_compute_deeponet.hessian(self.model,data, cuda=True)

        # todo: plot loss landscape
        # top_eigenvalues, top_eigenvector = hessian_comp.eigenvalues()

        def perturb_top_2_params(alpha=1e-3, beta=1e-3, top_eigenvector_1=None, top_eigenvector_2=None, original_params=None):

            # Perturb only the parameters corresponding to the top eigenvalues
            with torch.no_grad():
                param_list = list(original_params)

                # v1 = top_eigenvector_1
                # v2 = top_eigenvector_2

                pert_1 = []
                pert_2 = []

                for v1_layer, v2_layer, ori_layer in zip(top_eigenvector_1, top_eigenvector_2, param_list):
                    # Normalize each layer separately
                    # if v1_layer.ndimension() > 1:  # Only normalize multi-dimensional tensors (e.g., conv/linear weights)
                    norm_v1 = torch.norm(v1_layer, dim=list(range(1, v1_layer.ndimension())), keepdim=True) + 1e-20
                    norm_v2 = torch.norm(v2_layer, dim=list(range(1, v2_layer.ndimension())), keepdim=True) + 1e-20
                    norm_weights = torch.norm(ori_layer, dim=list(range(1, ori_layer.ndimension())), keepdim=True) + 1e-20
                    # perturb direction
                    v1_layer = v1_layer / norm_v1 * norm_weights
                    v2_layer = v2_layer / norm_v2 * norm_weights

                    pert_1.append(alpha * v1_layer)
                    pert_2.append(beta * v2_layer)

                # Apply perturbation
                param_pert = [p + dv1 + dv2 for p, dv1, dv2 in zip(param_list, pert_1, pert_2)]

                # norm_v1 = torch.norm(v1) + 1e-20
                # norm_v2 = torch.norm(v2) + 1e-20
                #
                # v1 = v1 / norm_v1
                # v2 = v2 / norm_v2
                #
                # pert_1 = [alpha * v for v in v1]
                # pert_2 = [beta * v for v in v2]
                # param_pert = pert_1+pert_2+param_list

            return param_pert

        def compute_loss():
            """
            Compute loss on a validation set (assumes loss function is defined).
            """
            self.model.eval()
            # with torch.no_grad():
            # loss = self.res_loss(t, pq0)


            loss_res1, loss_res2 = self.res_loss_sep(t, pq0)
            loss_ic1, loss_ic2 = self.ic_loss(t, pq0,pq0)


            # Compute final loss
            loss = res1_ws * loss_res1 + res2_ws * loss_res2 + ic1_ws * loss_ic1 + ic2_ws * loss_ic2

            # loss = pde1_loss+pde2_loss
            return loss
        def plot_loss_landscape( alpha_range=(-1, 1), beta_range=(-1, 1), steps=21):
            """
            Perturb the model in top-2 eigenvector directions and plot the loss landscape.
            """
            alphas = np.linspace(alpha_range[0], alpha_range[1], steps)
            betas = np.linspace(beta_range[0], beta_range[1], steps)
            loss_grid = np.zeros((steps, steps))
            original_params = [p.clone() for p in self.model.parameters()]
            eigenvalues, eigenvectors = hessian_comp.eigenvalues(top_n=2)

            # Find the indices of the top 2 eigenvalues
            sorted_indices = torch.argsort(torch.tensor(eigenvalues), descending=True)
            top_2_indices = sorted_indices[:2]

            # Select corresponding eigenvectors
            top_eigenvector_1 = eigenvectors[top_2_indices[0]]
            top_eigenvector_2 = eigenvectors[top_2_indices[1]]

            for i, alpha in enumerate(alphas):
                for j, beta in enumerate(betas):

                    # Apply perturbation
                    param_pert = perturb_top_2_params(alpha, beta, top_eigenvector_1, top_eigenvector_2,original_params)
                    with torch.no_grad():
                        for param, pert in zip(self.model.parameters(), param_pert):
                            param.copy_(pert)

                    loss_grid[i, j] = compute_loss()


            A, B = np.meshgrid(alphas, betas)
            with open(
                    f'/home/xinmeng/pinn_bow_mass/trained_model/fa25/export_figure/losslandscape_data/'
                    f'losslandscape_{self.Fb}_deeponet.pkl',
                    "wb") as f:
                pickle.dump((A, B, loss_grid), f)

            import matplotlib.ticker as mticker

            # My axis should display 10⁻¹ but you can switch to e-notation 1.00e+01
            def log_tick_formatter(val, pos=None):
                return f"$10^{{{int(val)}}}$"  # remove int() if you don't use MaxNLocator
                # return f"{10**val:.2e}"      # e-Notation

            fig = plt.figure(figsize=(5.5 * 2, 3.5 * 2))
            font = 20
            ax = fig.add_subplot(111, projection='3d')
            #
            cbar = fig.colorbar(ax.plot_surface(A, B,  np.log10(loss_grid), cmap="plasma"), ax=ax)
            cbar.ax.yaxis.set_major_formatter(mticker.FuncFormatter(log_tick_formatter))

            # Set colorbar label and tick properties
            cbar.set_label("Loss", fontsize=font)
            cbar.ax.tick_params(labelsize=font)
            ax.set_xlabel(r"$\varepsilon_1$", fontsize=font, labelpad=10)
            ax.set_ylabel(r"$\varepsilon_2$", fontsize=font, labelpad=10)
            ax.set_title(r"$F_B = {}$".format(self.Fb), fontsize=font)
            ax.zaxis.set_major_formatter(mticker.FuncFormatter(log_tick_formatter))
            ax.zaxis.set_major_locator(mticker.MaxNLocator(integer=True))
            ax.tick_params(axis='x', labelsize=font)
            ax.tick_params(axis='y', labelsize=font)
            ax.tick_params(axis='z', labelsize=font)
            ax.grid(False)
            ax.view_init(elev=30, azim=45)
            fig.tight_layout()
            plt.show()
            print('')


        plot_loss_landscape(alpha_range=(-0.5, 0.5), beta_range=(-0.5,0.5), steps=101) #ori:201




        # density_eigen, density_weight = hessian_comp.density()
        # # density_plot.get_esd_plot(density_eigen, density_weight)
        # # density, grids = density_plot.density_generate(density_eigen, density_weight,num_bins=10000)
        # density, grids = density_plot.density_generate(density_eigen, density_weight,num_bins=1000000)
        # with open(
        #         f'/home/xinmeng/pinn_bow_mass/trained_model/fa25/export_figure/hessian_eigen_data/'
        #         f'hessian_eigen_fb_{self.Fb}_deeponet.pkl',
        #         "wb") as f:
        #     pickle.dump((density_eigen, density_weight), f)
        # plt.plot(grids, density)
        # # plt.ylim(5e-8, 5e-7)
        # plt.yscale("log")
        # plt.xscale("log")
        # plt.ylabel('Density (Log Scale)', fontsize=14)
        # plt.xlabel('Eigenvlaue', fontsize=14)
        # plt.xticks(fontsize=12)
        # plt.yticks(fontsize=12)
        # plt.tight_layout()
        # plt.show()
        # plt.savefig(
        #     f'/home/xinmeng/pinn_bow_mass/trained_model/fa25/export_figure/hessian_eigen_deeponet_fb{self.Fb}.pdf')
        #
        return



    def test_anneal(self, model_path):
            self.pre_proc()
            self.p_vis = []
            self.q_vis = []
            checkpoint = torch.load(model_path)
            runtime = (checkpoint['elapsed_time']) / 60 / 60
            print(f'runtime: {runtime} hour')
            self.model.load_state_dict(checkpoint['model_state_dict'])
            # self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

            epoch = checkpoint['epoch']
            tot_losses = checkpoint['tot_losses']
            res1_losses = checkpoint['res1_losses']
            res2_losses = checkpoint['res2_losses']
            ic1_losses = checkpoint['ic1_losses']
            ic2_losses = checkpoint['ic2_losses']

            tot_losses = [loss for loss in tot_losses]
            res1_losses = [loss for loss in res1_losses]
            res2_losses = [loss for loss in res2_losses]
            ic1_losses = [loss for loss in ic1_losses]
            ic2_losses = [loss for loss in ic2_losses]

            # Plot loss
            plt.figure(figsize=(10, 6))

            plt.plot( tot_losses, label='Total Loss', color='blue')
            plt.plot( res1_losses, label='Residual Loss1', color='pink')
            plt.plot(res2_losses, label='Residual Loss2', color='red')
            plt.plot( ic1_losses, label='IC1 Loss', color='green')
            plt.plot( ic2_losses, label='IC2 Loss', color='orange')

            plt.yscale('log')
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.title('Training Losses Over Epochs')
            plt.legend()
            plt.grid(True)
            plt.show()

            self.Q = 500
            t = torch.linspace(0, self.time_length, self.Q).to(device).requires_grad_()
            t = t.unsqueeze(1)
            p0_value =  0
            q0_value = 0
            tmax = 0.5
            for time_seg in range(int(tmax/self.time_length)):
                self.optimizer.zero_grad()

                p0 = p0_value * torch.ones( self.Q, 1).to(device).requires_grad_()
                q0 = q0_value * torch.ones( self.Q, 1).to(device).requires_grad_()
                pq0 = torch.cat((p0, q0), dim=1)

                p, q = self.model(t, pq0)

                p0_value = p.squeeze()[-1]
                q0_value = q.squeeze()[-1]

                if time_seg != 0:
                    p = p[1:]
                    q = q[1:]
                self.p_vis.extend(p.squeeze().detach().cpu().tolist())
                self.q_vis.extend(q.squeeze().detach().cpu().tolist())

            self.load_fd_result()
            self.visualize(np.array(self.p_vis), np.array(self.q_vis), epoch, tmax=tmax)
            p_deeponet = np.array(self.p_vis)
            q_deeponet = np.array(self.q_vis)
            t_deeponet = np.linspace(0, tmax, np.size(q_deeponet))

            with open(
                    f'/home/xinmeng/pinn_bow_mass/trained_model/fa25/export_figure/'
                    f'fb_{self.Fb}_deeponet_hybrid.pkl',
                    "wb") as f:
                pickle.dump((t_deeponet, p_deeponet, q_deeponet), f)



    def nmse(self, p, q):
        # Compute MSE (Mean Squared Error)
        mse = np.mean((p - q) ** 2)

        # Compute the variance of p
        var_p = np.var(p)

        # Compute the NMSE
        nmse_value = mse / var_p
        return nmse_value

    def ncc(self, p, q):
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

    def test_generalize(self, model_path):
        self.pre_proc()
        self.p_vis = []
        self.q_vis = []
        checkpoint = torch.load(model_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        data_fd = loadmat(f'../data/gt_fd_deeponet_generalize_random_num100_fb_{self.Fb}.mat')

        p_fd = data_fd['p'].squeeze()
        q_fd = data_fd['q'].squeeze()
        fs = data_fd['fs'].squeeze()
        tot_time = data_fd['tot_time'].squeeze()

        N_data = np.size(p_fd,1)

        nmse_p = []
        nmse_q = []
        ncc_p = []
        ncc_q = []
        Q = int(fs*self.time_length) + 1
        for ii in range(N_data):
            p_deep = []
            q_deep = []

            # t = torch.rand(self.Q, 1).to(device).requires_grad_()
            p0_value = p_fd[0,ii]
            q0_value = q_fd[0,ii]

            for time_seg in range(int(tot_time / self.time_length)):
                self.optimizer.zero_grad()
                t = torch.linspace(0, self.time_length, Q).to(device).requires_grad_()
                t = t.unsqueeze(1)
                p0 = p0_value * torch.ones(Q, 1).to(device).requires_grad_()
                q0 = q0_value * torch.ones(Q, 1).to(device).requires_grad_()
                pq0 = torch.cat((p0, q0), dim=1)

                p, q = self.model(t, pq0)

                p = p.detach()
                q = q.detach()

                p0_value = p.squeeze()[-1]
                q0_value = q.squeeze()[-1]

                # if time_seg != 0:
                p = p[0:-1]
                q = q[0:-1]
                p_deep.extend(p.squeeze().detach().cpu().numpy().tolist())
                q_deep.extend(q.squeeze().detach().cpu().numpy().tolist())

            p_deep_np = np.array(p_deep)
            q_deep_np = np.array(q_deep)

            nmse_p.append(self.nmse( p_fd[:,ii].squeeze(), p_deep_np))
            nmse_q.append(self.nmse( q_fd[:,ii].squeeze(), q_deep_np))
            ncc_p.append(self.ncc( p_fd[:,ii].squeeze(), p_deep_np))
            ncc_q.append(self.ncc(q_fd[:,ii].squeeze(), q_deep_np))

        nmse_p_mean = np.mean(np.array(nmse_p))
        nmse_q_mean = np.mean(np.array(nmse_q))
        ncc_p_mean= np.mean(np.array(ncc_p))
        ncc_q_mean = np.mean(np.array(ncc_q))

        print(f'nmse_p:{nmse_p_mean}')
        print(f'nmse_q:{nmse_q_mean}')
        print(f'ncc_p:{ncc_p_mean}')
        print(f'ncc_q:{ncc_q_mean}')

        data = {
            'nmse_p_mean': nmse_p_mean,
            'nmse_q_mean': nmse_q_mean,
            'ncc_p_mean': ncc_p_mean,
            'ncc_q_mean': ncc_q_mean
        }

        # Specify the filename for saving
        filename = f'/home/xinmeng/pinn_bow_mass/trained_model/fa25/export_figure/deeponet_generalize_metrics_fb{self.Fb}.pkl'

        # Save using pickle
        with open(filename, 'wb') as f:
            pickle.dump(data, f)


def load_generalize(fb):
    filename = f'/home/xinmeng/pinn_bow_mass/trained_model/fa25/export_figure/deeponet_generalize_metrics_fb{fb}.pkl'

    # Load the pickle file
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    nmse_p_mean = data['nmse_p_mean']
    nmse_q_mean = data['nmse_q_mean']
    ncc_p_mean = data['ncc_p_mean']
    ncc_q_mean = data['ncc_q_mean']
    print(f'nmse_p:{nmse_p_mean}')
    print(f'nmse_q:{nmse_q_mean}')
    print(f'ncc_p:{ncc_p_mean}')
    print(f'ncc_q:{ncc_q_mean}')


if __name__ == "__main__":

    isvisual = False
    isfourier = True
    fb_value = 1000

    is_train = False
    is_hybrid = True
    if fb_value != 1000:
        is_hybrid = False
    is_test = True
    is_test_hessian = False
    is_test_generalize = False

     # ori: 0.01
    # load_generalize(fb_value)

    optimizer_type = 'soap' #adam
    if fb_value == 1000:
        time_length = 0.01 #todo: 0.001
        pq_max = 2
        fourier_sigma = 3
        layer_size_branch = [100, 100, 100, 100, 100, 100, 100, 200]
        layer_size_trunk = [100, 100, 100, 100, 100, 100, 100, 200]
    elif fb_value == 100:
        time_length = 0.01
        pq_max = 0.35
        fourier_sigma = 1
        layer_size_branch = [100, 100, 100, 100, 100, 100, 100, 200]
        layer_size_trunk = [100, 100, 100, 100, 100, 100, 100, 200]
    elif fb_value == 10:
        time_length = 0.01
        pq_max = 0.35
        fourier_sigma = 1
        layer_size_branch = [100, 100, 100, 100, 100, 100, 100, 200]
        layer_size_trunk = [100, 100, 100, 100, 100, 100, 100, 200]
    trainer = Trainer(isvisual=isvisual, isfourier=isfourier, fb_value=fb_value, pq_max=pq_max,
                      time_length=time_length, fourier_sigma = fourier_sigma,
                      layer_size_branch= layer_size_branch, layer_size_trunk= layer_size_trunk, optimizer_type = optimizer_type)
    root_path = "../saved_data"

    if is_train:
        if is_hybrid:
            trainer.train_lr_anneal_hybrid()
        else:
            trainer.train_lr_anneal()

    if is_test or is_test_hessian or is_test_generalize:

        if fb_value == 10:
            test_anneal_path = ("dafx_lr_anneal_soap_oneds_deeponet_bowmass_fb10_timelength_0.01_isfourier_True_sigma_1_itr250000.pth")
        elif fb_value == 100:
            test_anneal_path = ("dafx_lr_anneal_soap_oneds_deeponet_bowmass_fb100_timelength_0.01_isfourier_True_sigma_1_itr150000.pth")
        elif fb_value == 1000:
            if is_hybrid:
                test_anneal_path = ("dafx_hybrid_lr_anneal_soap_oneds_deeponet_bowmass_fb1000_timelength_0.01_isfourier_True_sigma_3_itr300000.pth")
            else:
                test_anneal_path = ("dafx_lr_anneal_soap_oneds_deeponet_bowmass_fb1000_timelength_0.01_isfourier_True_sigma_3_itr100000.pth")

        if is_test:
            trainer.test_anneal(os.path.join(root_path, "trained_model/deeponet",test_anneal_path))
        if is_test_hessian:
            trainer.test_hessian(os.path.join(root_path, "trained_model/deeponet",test_anneal_path))
        if is_test_generalize:
            trainer.test_generalize(os.path.join(root_path, "trained_model/deeponet",test_anneal_path))


