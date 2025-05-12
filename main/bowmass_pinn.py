'''
Worked hyperparameters:
2nd order:
    utt + omega^2 u + Fb sqrt(2a) (yt-vb) e^(-a (ut-vb)^2 + 1/2)= 0
1sr order:
    omega ut = omega ^2 u
    utt = - omega ut
or
    qt = omega p
    pt = - omega q - Fb sqrt(2a) (yt-vb) e^(-a (ut-vb)^2 + 1/2)
------------------------------
f = 1
omega = 2 pi f
T = 1
q = omega u
p = ut
------------------------------
exact solution:
u_ex = sin(omega t)
p_ex = omega cos(omega t)
q_ex = omega sin(omega t)
pt_ex = - omega^2 sin(omega t)
qt_ex = omega^2 cos(omega t)

TODO: T = 0.015 trained fast, T=0.03 kinda failed: since the gradient is too big
'''
import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
sys.path.append('/home/xinmeng/pinn_bow_mass')
import bow_main.network.nn_mod_mlp as NN
import bow_main.data.para_bow as para
from bow_main.FA25_bowmass.soap import SOAP
from datetime import datetime
from scipy.io import loadmat
import time
from tqdm import tqdm
import pickle
from torch.autograd.functional import hessian

os.environ["LD_LIBRARY_PATH"] = "/home/xinmeng/miniconda3/envs/thesis/lib:/nas/home/xluan/miniconda3/envs/thesis/lib/cuda/lib64"
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
    def __init__(self, isvisual, T, Fb, pqmax, optimzer_type,
                 isanneal, iscausal, p_ic, q_ic, nn_num, sigma):
        self.starter_learning_rate = 1e-3
        self.decay_rate = 0.9
        self.decay_steps = 10000 #ori: 1000
        self.n_epochs = 2000002
        self.omega = 2 * np.pi * 100
        self.T = T
        self.Fb = Fb
        self.pqmax = pqmax
        self.isanneal = isanneal
        self.optimizer_type = optimzer_type
        self.tot_losses = []
        self.pde_losses = []
        self.ic1_losses = []
        self.ic2_losses = []
        self.pde1_losses = []
        self.pde2_losses = []
        self.pde1_ws = []
        self.pde2_ws = []
        self.ic1_ws = []
        self.ic2_ws = []
        self.isvisual = isvisual
        self.chunk = 5*10
        self.batch_size = 1000
        self.causal_eta  = 1e-4
        self.iscausl = iscausal
        self.p_ic = p_ic
        self.q_ic = q_ic
        self.nn_num = nn_num
        self.sigma = sigma

    def load_fd_result(self):
        p_q = loadmat(f'../data/p_q_fd_05_fb_{self.Fb}.mat')

        # p_q = loadmat('../data/p_q_fd_01.mat')
        self.p_fd = p_q['p'].squeeze()
        self.q_fd = p_q['q'].squeeze()
        self.t_p_fd = p_q['t_p'].squeeze()
        self.t_q_fd = p_q['t_q'].squeeze()

    def pre_proc(self):
        layer_size = [100, 100, 100, 100, 100, 1] # TODO: 2
        self.model = NN.ModifiedMLP(layer_size, T=self.T, pqmax=self.pqmax, sigma=self.sigma).to(device)
        N_total_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Total trainable parameters: {N_total_params}")

        lambda_lr = lambda step: self.decay_rate ** (step / self.decay_steps)
        if optimizer_type == 'soap':
            self.optimizer = SOAP(self.model.parameters(), lr=3e-3, betas=(.95, .95), weight_decay=.01, precondition_frequency=10)
        elif optimizer_type == 'adam':
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.starter_learning_rate)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lambda_lr)
        self.start_time = time.time()
        if self.isvisual:
            self.load_fd_result()

    # Define the physics-informed loss (PDE residual)
    def pde_loss(self, t):

        p, q = self.model(t)
        # qt = omega pD
        # pt = - omega q - Fb sqrt(2a) (ut-vb) e^(-a (p-vb)^2 + 1/2)
        vb = 0.2  # [m/s]
        Fb = self.Fb  # 100, 4000
        a = 100
        eta = p - vb
        phi = torch.sqrt(2 * torch.tensor(a)) * torch.exp(-a * (eta ** 2) + 0.5) * eta

        p_t = torch.autograd.grad(p, t, grad_outputs=torch.ones_like(p), create_graph=True)[0]
        q_t = torch.autograd.grad(q, t, grad_outputs=torch.ones_like(q), create_graph=True)[0]

        pde_residual1 = p_t + self.omega * q + Fb * phi
        pde_residual2 = q_t - self.omega * p
        pde1_loss = torch.mean((pde_residual1 - 0) ** 2)
        pde2_loss = torch.mean((pde_residual2 - 0) ** 2)
        # pde_loss = pde1_loss + pde2_loss

        return pde1_loss, pde2_loss, p, q



    def pde_loss_hessian(self, t):
        def get_pq(t):
            return torch.cat(self.model(t), dim=1)

        hessian = torch.autograd.functional.hessian(get_pq, t)
        norm = torch.norm(hessian, p='fro')

        return  norm

    # Initial condition loss
    def initial_condition_loss(self):
        t0 = torch.tensor([[0.0]], requires_grad=True).to(device)
        p0, q0 = self.model(t0)

        # p_ex = omega cos(omega t)
        # q_ex = omega sin(omega t)
        ic_loss1 = (torch.mean((p0 - self.p_ic) ** 2))
        ic_loss2 = torch.mean((q0 - self.q_ic) ** 2)
        # ic_loss2 = torch.mean((u0_t - 0) ** 2)
        return ic_loss1, ic_loss2

    def save_model(self, model, optimizer, epoch, tot_losses, pde1_losses,pde2_losses, ic1_losses, ic2_losses, path):
        end_time = time.time()
        elapsed_time = end_time - self.start_time

        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch': epoch,
            'tot_losses': tot_losses,
            'pde1_losses': pde1_losses,
            'pde2_losses': pde2_losses,
            'ic1_losses': ic1_losses,
            'ic2_losses': ic2_losses,
            'elapsed_time': elapsed_time

        }, path)

    def visualize(self, t_train, p_out, q_out, epoch):

        fig, ax = plt.subplots(2, 1, figsize=(8, 6))
        t_train = t_train.detach().cpu().numpy()
        # t_train = t_train + np.max(t_train) * (self.nn_num-1) #todo
        t_train = t_train + 0.023
        # ax[0].plot(t_train.detach().numpy(), p_exact, label='Exact Solution')
        ax[0].plot(t_train, p_out.detach().cpu().numpy(), label='PINN', linestyle=':')
        ax[0].plot(self.t_p_fd, self.p_fd, label='FD', linestyle='-')
        ax[0].set_xlabel('Time [s]')
        ax[0].set_ylabel('p (t)')
        ax[0].set_xlim([np.min(t_train), np.max(t_train)])
        # ax[0].set_ylim([0.1, 0.4])
        ax[0].legend()
        ax[0].set_title(f'p: epoch {epoch}')

        # ax[1].plot(t_train.detach().numpy(), q_exact, label='Exact Solution')
        ax[1].plot(t_train, q_out.detach().cpu().numpy(), label='PINN', linestyle=':')
        ax[1].plot(self.t_q_fd, self.q_fd, label='FD', linestyle='-')
        ax[1].set_xlabel('Time [s]')
        ax[1].set_ylabel('q (t)')
        # ax[1].set_ylim([0.01, 1])
        ax[1].set_xlim([np.min(t_train), np.max(t_train)])
        ax[1].legend()
        ax[1].set_title(f'q: epoch {epoch}')

        # Adjust layout and show the figure
        plt.tight_layout()
        plt.show()
        print('')
    def train(self):
        self.pre_proc()
        t_train = torch.linspace(0, self.T, 1000).view(-1, 1).requires_grad_(True).to(device)  # Time domain [0,1]

        for epoch in tqdm(range(self.n_epochs), desc="Training Epochs"):
            self.optimizer.zero_grad()

            # PDE loss
            loss_pde1, loss_pde2, p_out, q_out = self.pde_loss(t_train)
            # loss_pde = loss_pde1+loss_pde2

            # Initial condition loss
            loss_ic1, loss_ic2 = self.initial_condition_loss()

            # Total loss
            # loss_pde = loss_pde *1e1#/ (0.03**2)
            # loss_ic1 = loss_ic1 * 1e6#/ 1e-8
            # loss_ic2 = loss_ic2 * 1e6#/ 1e-8

            loss = loss_pde1 * 1e1 + loss_pde2 * 1e1 + loss_ic1 * 1e6 + loss_ic2 * 1e6
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()

            self.tot_losses.append(loss.detach().cpu().item())
            self.pde1_losses.append(loss_pde1.detach().cpu().item())
            self.pde2_losses.append(loss_pde1.detach().cpu().item())
            self.ic1_losses.append(loss_ic1.detach().cpu().item())
            self.ic2_losses.append(loss_ic2.detach().cpu().item())

            if epoch % 1000 == 0:
                print(f'Epoch {epoch}, loss_pde1: {loss_pde1}, loss_pde2: {loss_pde2}, loss_ic1:{loss_ic1}, loss_ic2:{loss_ic2}')
                if self.isvisual:
                    self.visualize(t_train, p_out, q_out, epoch)



            if (epoch+1) % (20000) == 0:
                save_path = (f'/home/xinmeng/pinn_bow_mass/trained_model/fa25/pinn/'
                             f'bow_mass_model_modmlp_Fb_anneal_{self.isanneal}_{self.Fb}_{self.optimizer_type}_{self.T}s_{epoch + 1}_nn_num{self.nn_num}.pth')
                self.save_model(self.model, self.optimizer, self.n_epochs, self.tot_losses, self.pde1_losses,self.pde2_losses,
                                self.ic1_losses, self.ic2_losses, save_path)
                print(f'Model saved at epoch: {epoch + 1}.')

        return self.model
    def loss_annealing(self, loss_res1, loss_res2, loss_ic1, loss_ic2):
        # Compute gradient norms for balancing
        loss_res1_norm = torch.autograd.grad(loss_res1, self.model.nn1.parameters(), retain_graph=True, allow_unused=True)
        loss_res2_norm = torch.autograd.grad(loss_res2, self.model.nn2.parameters(), retain_graph=True, allow_unused=True)
        loss_ic1_norm = torch.autograd.grad(loss_ic1, self.model.nn1.parameters(), retain_graph=True, allow_unused=True)
        loss_ic2_norm = torch.autograd.grad(loss_ic2, self.model.nn2.parameters(), retain_graph=True, allow_unused=True)

        loss_res1_norm = sum(p.norm(2) for p in loss_res1_norm if p is not None)
        loss_res2_norm = sum(p.norm(2) for p in loss_res2_norm if p is not None)
        loss_ic1_norm = sum(p.norm(2) for p in loss_ic1_norm if p is not None)
        loss_ic2_norm = sum(p.norm(2) for p in loss_ic2_norm if p is not None)

        # Compute dynamic weights
        total_norm = loss_res1_norm + loss_res2_norm + loss_ic1_norm + loss_ic2_norm
        w_res1 = total_norm / (loss_res1_norm + 1e-20)
        w_res2 = total_norm / (loss_res2_norm + 1e-20)
        w_ic1 = total_norm / (loss_ic1_norm + 1e-20)
        w_ic2 = total_norm / (loss_ic2_norm + 1e-20)

        return w_res1, w_res2, w_ic1, w_ic2

    def save_model_anneal(self, model, optimizer, epoch, tot_losses, pde1_losses, pde2_losses,
                          ic1_losses, ic2_losses,
                          pde1_ws, pde2_ws, ic1_ws, ic2_ws,path):
        end_time = time.time()
        elapsed_time = end_time - self.start_time

        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch': epoch,
            'tot_losses': tot_losses,
            'pde1_losses': pde1_losses,
            'pde2_losses': pde2_losses,
            'ic1_losses': ic1_losses,
            'ic2_losses': ic2_losses,
            'pde1_ws': pde1_ws,
            'pde2_ws': pde2_ws,
            'ic1_ws': ic1_ws,
            'ic2_ws': ic2_ws,
            'elapsed_time': elapsed_time

        }, path)

    def train_lr_anneal(self):
        self.pre_proc()

        t_train = torch.linspace(0, self.T, 5000).view(-1, 1).requires_grad_(True).to(device)  # Time domain [0,1]

        for epoch in tqdm(range(self.n_epochs), desc="Training Epochs"):

            self.optimizer.zero_grad()

            # PDE loss
            loss_pde1, loss_pde2, p_out, q_out = self.pde_loss(t_train)

            # Initial condition loss
            loss_ic1, loss_ic2 = self.initial_condition_loss()


            if epoch % 1000 == 0:
                w_res1, w_res2, w_ic1, w_ic2 = self.loss_annealing(loss_pde1, loss_pde2, loss_ic1, loss_ic2)

            loss = w_res1 * loss_pde1 +  w_res2 * loss_pde2+ w_ic1 * loss_ic1 + w_ic2 * loss_ic2
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()

            self.tot_losses.append(loss.detach().cpu().item())
            self.pde1_losses.append(loss_pde1.detach().cpu().item())
            self.pde2_losses.append(loss_pde2.detach().cpu().item())
            self.ic1_losses.append(loss_ic1.detach().cpu().item())
            self.ic2_losses.append(loss_ic2.detach().cpu().item())
            self.pde1_ws.append(w_res1.detach().cpu().item())
            self.pde2_ws.append(w_res2.detach().cpu().item())
            self.ic1_ws.append(w_ic1.detach().cpu().item())
            self.ic2_ws.append(w_ic2.detach().cpu().item())

            if epoch % 1000 == 0:
                print(f'Epoch {epoch}, loss_pde1: {loss_pde1}, loss_pde2: {loss_pde2}, '
                      f'loss_ic1:{loss_ic1}, loss_ic2:{loss_ic2},'
                      f'w_pde1:{w_res1}, w_pde2: {w_res2}, w_ic1: {w_ic1}, w_ic2: {w_ic2}')
                if self.isvisual:
                    self.visualize(t_train, p_out, q_out, epoch)

            if (epoch + 1) % (5000) == 0:
                save_path = (f'/home/xinmeng/pinn_bow_mass/trained_model/fa25/pinn/'
                             f'bow_mass_model_modmlp_Fb_anneal_{self.isanneal}_{self.Fb}_{self.optimizer_type}_{self.T}s_{epoch + 1}.pth')
                self.save_model_anneal(self.model, self.optimizer, self.n_epochs, self.tot_losses, self.pde1_losses, self.pde2_losses,
                                self.ic1_losses, self.ic2_losses, self.pde1_ws, self.pde2_ws, self.ic1_ws, self.ic2_ws, save_path)
                print(f'Model saved at epoch: {epoch + 1}.')

        return self.model


    def train_lr_anneal_casual(self):
        self.pre_proc()

        t_train = torch.linspace(0, self.T/self.chunk, 1000).view(-1, 1).requires_grad_(True).to(device)  # Time domain [0,1]
        i_chunk = 0
        for epoch in tqdm(range(self.n_epochs), desc="Training Epochs"):

            self.optimizer.zero_grad()

            # PDE loss
            loss_pde1, loss_pde2, p_out, q_out = self.pde_loss(t_train)

            # Initial condition loss
            loss_ic1, loss_ic2 = self.initial_condition_loss()

            if epoch % 1000 == 0:
                w_res1, w_res2, w_ic1, w_ic2 = self.loss_annealing(loss_pde1, loss_pde2, loss_ic1, loss_ic2)


            loss = (w_res1 * loss_pde1 +
                    w_res2 * loss_pde2 +
                    w_ic1 * loss_ic1 +
                    w_ic2 * loss_ic2
                 )

            loss.backward()
            self.optimizer.step()
            self.scheduler.step()
            #5
            if loss_pde1 < 0.1  and i_chunk < self.chunk: #todo: fb=1000: 0.1
                save_path = (f'/home/xinmeng/pinn_bow_mass/trained_model/fa25/pinn/'
                             f'bow_mass_model_modmlp_Fb_iscausal_{self.iscausl}_ichunk_{i_chunk}_anneal_{self.isanneal}_{self.Fb}_{self.optimizer_type}_{self.T}s_{epoch + 1}_N1000_nn_{self.nn_num}.pth')

                self.save_model_anneal(self.model, self.optimizer, self.n_epochs, self.tot_losses, self.pde1_losses,
                                       self.pde2_losses,
                                       self.ic1_losses, self.ic2_losses, self.pde1_ws, self.pde2_ws, self.ic1_ws,
                                       self.ic2_ws, save_path)
                print(f'Model saved at epoch: {epoch + 1}.')
                i_chunk = i_chunk+1
                # t_train = torch.linspace(0, self.T / self.chunk * (i_chunk+1), 1000 * (i_chunk+1)).view(-1, 1).requires_grad_(True).to(
                #     device)
                t_train = torch.linspace(0, self.T / self.chunk * (i_chunk+1), 1000 ).view(-1, 1).requires_grad_(True).to(
                    device)
                print(f'i_chunk:{i_chunk}')

            self.tot_losses.append(loss.detach().cpu().item())
            # self.pde1_losses.append(sum_loss_pde1[-1].detach().cpu().item())
            # self.pde2_losses.append(sum_loss_pde2[-1].detach().cpu().item())
            self.pde1_losses.append(loss_pde1.detach().cpu().item())
            self.pde2_losses.append(loss_pde2.detach().cpu().item())
            self.ic1_losses.append(loss_ic1.detach().cpu().item())
            self.ic2_losses.append(loss_ic2.detach().cpu().item())
            self.pde1_ws.append(w_res1.detach().cpu().item())
            self.pde2_ws.append(w_res2.detach().cpu().item())
            self.ic1_ws.append(w_ic1.detach().cpu().item())
            self.ic2_ws.append(w_ic2.detach().cpu().item())

            if epoch % 1000 == 0:
                print(f'Epoch {epoch}, loss_pde1: {loss_pde1}, loss_pde2: {loss_pde2}, '
                      f'loss_ic1:{loss_ic1}, loss_ic2:{loss_ic2},'
                      f'w_pde1:{w_res1}, w_pde2: {w_res2}, w_ic1: {w_ic1}, w_ic2: {w_ic2}')

            if (epoch + 1) % (10000) == 0:
                save_path = (f'/home/xinmeng/pinn_bow_mass/trained_model/fa25/pinn/'
                             f'bow_mass_model_modmlp_Fb_iscausal_{self.iscausl}_ichunk_{i_chunk}_anneal_{self.isanneal}_{self.Fb}_{self.optimizer_type}_{self.T}s_{epoch + 1}_N1000_nn_{self.nn_num}.pth')
                self.save_model_anneal(self.model, self.optimizer, self.n_epochs, self.tot_losses, self.pde1_losses,
                                       self.pde2_losses,
                                       self.ic1_losses, self.ic2_losses, self.pde1_ws, self.pde2_ws, self.ic1_ws,
                                       self.ic2_ws, save_path)
                print(f'Model saved at epoch: {epoch + 1}.')

        return self.model

    def test_hessian(self, model_path):
        self.pre_proc()
        checkpoint = torch.load(model_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        if fb_value == 1000:
            res1_ws = checkpoint['pde1_ws'][-1]
            res2_ws = checkpoint['pde2_ws'][-1]
            ic1_ws = checkpoint['ic1_ws'][-1]
            ic2_ws = checkpoint['ic2_ws'][-1]
        else:
            res1_ws = 1e1
            res2_ws = 1e1
            ic1_ws = 1e6
            ic2_ws = 1e6

        if self.Fb == 1000:
            if self.nn_num == 1:
                t_max = self.T
            elif self.nn_num == 2:
                t_max = self.T / 50 * 35
            elif self.nn_num == 3:
                t_max = self.T / 50 *16
            elif self.nn_num == 4:
                t_max = self.T / 50 * 14
            elif self.nn_num == 5:
                t_max = self.T / 50 * 23
        else:
            t_max = self.T
        t_train = torch.linspace(0, t_max, 5000).view(-1, 1).requires_grad_(True).to(
            device)  # Time domain [0,1]
        self.optimizer.zero_grad()
        import hessian_compute_pinn
        import density_plot
        hessian_comp = hessian_compute_pinn.hessian(self.model,t_train, cuda=True)
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
            loss_res1, loss_res2, _, _ = self.pde_loss(t_train)
            # loss_res1, loss_res2 = self.res_loss_sep(t, pq0)
            loss_ic1, loss_ic2 = self.initial_condition_loss()


            # Compute final loss
            loss = res1_ws * loss_res1 + res2_ws * loss_res2 + ic1_ws * loss_ic1 + ic2_ws * loss_ic2

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
            from matplotlib.ticker import ScalarFormatter

            # loss_grid = np.log10(loss_grid)

            A, B = np.meshgrid(alphas, betas)
            with open(
                    f'/home/xinmeng/pinn_bow_mass/trained_model/fa25/export_figure/losslandscape_data/'
                    f'losslandscape_{self.Fb}_pinn_nn{self.nn_num}.pkl',
                    "wb") as f:
                pickle.dump((A, B, loss_grid), f)

            # fig = plt.figure(figsize=(5.5*2, 3.5*2))
            # font = 20
            # ax = fig.add_subplot(111, projection='3d')
            # # ax.plot_surface(A, B, (loss_grid), cmap="plasma", edgecolor="none", alpha=0.5)
            # # surf = ax.plot_surface(A, B, loss_grid, cmap="plasma", edgecolor="none", alpha=0.5)
            # # cbar = fig.colorbar(surf, ax=ax, label="Loss")
            # # cbar.ax.tick_params(labelsize=font)  # 设置 colorbar 字体大小
            # # cbar.set_label("Loss", fontsize=font)
            #
            # cbar = fig.colorbar(ax.plot_surface(A, B, loss_grid, cmap="plasma"), ax=ax)
            # cbar.ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
            # cbar.ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
            # cbar.ax.yaxis.offsetText.set_fontsize(font)
            # cbar.set_label("Loss", fontsize=font)
            # cbar.ax.tick_params(labelsize=font)
            # # fig.colorbar(ax.plot_surface(A, B, (loss_grid), cmap="plasma"), ax=ax, label="Loss")
            # # ax.plot_surface(A, B, np.log10(loss_grid.T), cmap="plasma", edgecolor="none", alpha=0.1)
            # # fig.colorbar(ax.plot_surface(A, B, np.log10(loss_grid.T), cmap="plasma"), ax=ax, label="Loss")
            # # ax.set_zlim(-3, 9)
            # ax.set_xlabel(r"$\varepsilon_1$", fontsize=font, labelpad=10)
            # ax.set_ylabel(r"$\varepsilon_2$", fontsize=font, labelpad=10)
            # # ax.set_zlabel("Loss", fontsize=font)
            # ax.set_title(r"$F_B = {}$".format(self.Fb), fontsize=font)
            # ax.tick_params(axis='x', labelsize=font)
            # ax.tick_params(axis='y', labelsize=font)
            # # ax.tick_params(axis='z', labelsize=font)
            # ax.tick_params(axis='z', which='both', direction='in', labelsize=font)
            # ax.zaxis.set_major_formatter(ScalarFormatter(useMathText=True))  # 使用数学符号
            # ax.ticklabel_format(axis='z', style='sci', scilimits=(0, 0))  # 强制使用科学计数法
            # # ax.set_title("3D Loss Landscape")
            # ax.grid(False)
            # ax.view_init(elev=30, azim=45)
            # fig.tight_layout()
            # # ax.zlim(-2, 5)
            # # ax.set_zscale('log')
            # plt.savefig(
            #     f'/home/xinmeng/pinn_bow_mass/trained_model/fa25/export_figure/losslandscape_pinn_{self.Fb}_nn{self.nn_num}.pdf')
            # plt.show()
            # print('')

        plot_loss_landscape(alpha_range=(-0.5, 0.5), beta_range=(-0.5, 0.5), steps=101)  # ori:201

        # # hessian eigen value density
        # density_eigen, density_weight = hessian_comp.density()
        # # density_plot.get_esd_plot(density_eigen, density_weight)
        # density, grids = density_plot.density_generate(density_eigen, density_weight,num_bins=100000)
        # with open(
        #         f'/home/xinmeng/pinn_bow_mass/trained_model/fa25/export_figure/hessian_eigen_data/'
        #         f'hessian_eigen_fb_{self.Fb}_pinn_nn{self.nn_num}.pkl',
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
        # plt.savefig(f'/home/xinmeng/pinn_bow_mass/trained_model/fa25/export_figure/hessian_eigen_pinn_fb{self.Fb}_nn{self.nn_num}.pdf')
        # print('')


    def test(self, model_path):
        self.pre_proc()
        checkpoint = torch.load(model_path)
        runtime = (checkpoint['elapsed_time'])/60/60
        print(f'runtime: {runtime} hour')
        self.model.load_state_dict(checkpoint['model_state_dict'])
        # self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.model.eval()
        epoch = checkpoint['epoch']
        tot_losses = checkpoint['tot_losses']
        pde1_losses = checkpoint['pde1_losses']
        pde2_losses = checkpoint['pde2_losses']
        ic1_losses = checkpoint['ic1_losses']
        ic2_losses = checkpoint['ic2_losses']

        # pde1_ws = checkpoint['pde1_ws']
        # pde2_ws = checkpoint['pde2_ws']
        # ic1_ws = checkpoint['ic1_ws']
        # ic2_ws = checkpoint['ic2_ws']

        tot_losses = [loss for loss in tot_losses]
        pde1_losses = [loss for loss in pde1_losses]
        pde2_losses = [loss for loss in pde2_losses]
        ic1_losses = [loss for loss in ic1_losses]
        ic2_losses = [loss for loss in ic2_losses]
        # pde1_ws = [loss for loss in pde1_ws]
        # pde2_ws = [loss for loss in pde2_ws]
        # ic1_ws = [loss for loss in ic1_ws]
        # ic2_ws = [loss for loss in ic2_ws]

        # Plot loss
        plt.figure(figsize=(10, 6))

        plt.plot( tot_losses, label='Total Loss', color='blue')
        plt.plot( pde1_losses, label='PDE1 Loss', color='red')
        plt.plot(pde2_losses, label='PDE2 Loss', color='purple')
        plt.plot( ic1_losses, label='IC1 Loss', color='green')
        plt.plot( ic2_losses, label='IC2 Loss', color='orange')

        plt.yscale('log')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Training Losses Over Epochs')
        plt.legend()
        plt.grid(True)
        plt.show()

        # plt.figure(figsize=(10, 6))
        #
        # plt.plot( pde1_ws, label='PDE1 Loss', color='red')
        # plt.plot(pde2_ws, label='PDE2 Loss', color='purple')
        # plt.plot( ic1_ws, label='IC1 Loss', color='green')
        # plt.plot( ic2_ws, label='IC2 Loss', color='orange')
        #
        # plt.yscale('log')
        # plt.xlabel('Epochs')
        # plt.ylabel('Loss weights')
        # plt.title('Training Losses Over Epochs')
        # plt.legend()
        # plt.grid(True)
        # plt.show()


        if self.Fb == 1000:
            if self.nn_num == 1:
                t_max = self.T
            elif self.nn_num == 2:
                t_max = self.T / 50 * 35
            elif self.nn_num == 3:
                t_max = self.T / 50 *16
            elif self.nn_num == 4:
                t_max = self.T / 50 * 14
            elif self.nn_num == 5:
                t_max = self.T / 50 * 23
        else:
            t_max = self.T
        t_train = torch.linspace(0, t_max, 5000).view(-1, 1).requires_grad_(True).to(
            device)  # Time domain [0,1]

        # t_train = torch.linspace(0, self.T/50 *23, 5000).view(-1, 1).requires_grad_(True).to(device)  # Time domain [0,1]

        # PDE loss
        loss_pde1, loss_pde2, p_out, q_out = self.pde_loss(t_train)

        # plot result
        self.load_fd_result()
        self.visualize(t_train, p_out, q_out, epoch)

        if self.nn_num == 1:
            p_pinn = p_out.squeeze().detach().cpu().numpy()
            q_pinn = q_out.squeeze().detach().cpu().numpy()
        else:
            p_pinn = p_out.squeeze().detach().cpu().numpy()[1:]
            q_pinn = q_out.squeeze().detach().cpu().numpy()[1:]
        if self.Fb == 1000:
            if self.nn_num == 1:
                t_pinn = np.linspace(0, 0.01, np.size(q_pinn))
            elif self.nn_num == 2:
                t_pinn = np.linspace(0.01, 0.0170, np.size(q_pinn))
            elif self.nn_num == 3:
                t_pinn = np.linspace(0.0170, 0.0202, np.size(q_pinn))
            elif self.nn_num == 4:
                t_pinn = np.linspace(0.0202, 0.023, np.size(q_pinn))
            elif self.nn_num == 5:
                t_pinn = np.linspace(0.023, 0.0276, np.size(q_pinn))
        else:
            t_pinn = np.linspace(self.T * (self.nn_num - 1), self.T * self.nn_num, np.size(q_pinn))

        import pickle
        with open(
                f'/home/xinmeng/pinn_bow_mass/trained_model/fa25/export_figure/'
                f'fb_{self.Fb}_pinn_nn{self.nn_num}.pkl',
                "wb") as f:
            pickle.dump((t_pinn, p_pinn, q_pinn), f)

        # with open(
        #         "/home/xinmeng/pinn_bow_mass/trained_model/fa25/pinn/"
        #         "p_q_out_last_bow_mass_model_modmlp_Fb_iscausal_True_ichunk_15_anneal_True_1000_soap_0.01s_128474_N1000_nn_4.pkl",
        #         "wb") as f:
        #     pickle.dump((p_out.squeeze().detach().cpu().numpy()[-1], q_out.squeeze().detach().cpu().numpy()[-1]), f)
        # print('')
print('start....')
# Train the model
isvisual = False

fb_value = 1000
nn_num = 5
# time_length = 0.03 # ori: 0.01
if fb_value == 1000:
    pqmax = 1 # ori: 0.2
    T = 0.01
    isanneal = True
    iscausal = True
    sigma = 3
elif fb_value == 100:
    T = 0.03
    pqmax = 0.2  # ori: 0.2
    isanneal = False
    iscausal = False
    sigma = 1
elif fb_value == 10:
    T = 0.1
    pqmax = 0.2  # ori: 0.2
    isanneal = False
    iscausal = False
    sigma = 1
optimizer_type = 'soap' #adam
root_path = "/home/xinmeng/pinn_bow_mass/trained_model/fa25/pinn"


if nn_num == 2:
    if fb_value == 10:
        p_q_load_file = "p_q_out_last_bow_mass_model_modmlp_Fb_anneal_False_10_soap_0.1s_1000000.pkl"
    elif fb_value == 100:
        p_q_load_file = "p_q_out_last_bow_mass_model_modmlp_Fb_anneal_False_100_soap_0.03s_1000000.pkl"
    elif fb_value == 1000:
        p_q_load_file = "p_q_out_last_bow_mass_model_modmlp_Fb_iscausal_True_ichunk_50_anneal_True_1000_soap_0.01s_800000_N1000_nn_1.pkl"
elif nn_num == 3:
    if fb_value == 10:
        p_q_load_file = "p_q_out_last_bow_mass_model_modmlp_Fb_anneal_False_10_soap_0.1s_1000000_nn2.pkl"
    elif fb_value == 100:
        p_q_load_file = "p_q_out_last_bow_mass_model_modmlp_Fb_anneal_False_100_soap_0.03s_1000000_nn2.pkl"
    elif fb_value == 1000:
        p_q_load_file = "p_q_out_last_bow_mass_model_modmlp_Fb_iscausal_True_ichunk_34_anneal_True_1000_soap_0.01s_86279_N1000_nn_2.pkl"
elif nn_num == 4:
    if fb_value == 10:
        print("Warning: No exist")
        sys.exit()
    elif fb_value == 100:
        print("Warning: No exist")
        sys.exit()
    elif fb_value == 1000:
        p_q_load_file = "p_q_out_last_bow_mass_model_modmlp_Fb_iscausal_True_ichunk_15_anneal_True_1000_soap_0.01s_47963_N1000_nn_3.pkl"
elif nn_num == 5:
    if fb_value == 10:
        print("Warning: No exist")
        sys.exit()
    elif fb_value == 100:
        print("Warning: No exist")
        sys.exit()
    elif fb_value == 1000:
        p_q_load_file = "p_q_out_last_bow_mass_model_modmlp_Fb_iscausal_True_ichunk_15_anneal_True_1000_soap_0.01s_128474_N1000_nn_4.pkl"


if nn_num == 1:
    p_ic =0
    q_ic = 0
else:
    p_q_load_path = os.path.join(root_path, p_q_load_file)
    with open(p_q_load_path, "rb") as f:
        data = pickle.load(f)
    p_ic = data[0]
    q_ic = data[1]

# # ##########train
# # # #
trainer = Trainer(isvisual=isvisual, T=T, Fb = fb_value, pqmax =pqmax, isanneal=isanneal, optimzer_type=optimizer_type,
                  iscausal = iscausal, p_ic=p_ic, q_ic=q_ic, nn_num = nn_num, sigma=sigma)
# if iscausal:
#     trainer.train_lr_anneal_casual()
# elif  isanneal:
#     trainer.train_lr_anneal()
# else:
#     trainer.train()

# ##########test
# p_ic = 0
# q_ic =0

if iscausal:
    if nn_num == 1:
        test_file = "bow_mass_model_modmlp_Fb_iscausal_True_ichunk_50_anneal_True_1000_soap_0.01s_800000_N1000_nn_1.pth"
    elif nn_num == 2:
        test_file = "bow_mass_model_modmlp_Fb_iscausal_True_ichunk_34_anneal_True_1000_soap_0.01s_86279_N1000_nn_2.pth"
    elif nn_num == 3:
        test_file = "bow_mass_model_modmlp_Fb_iscausal_True_ichunk_15_anneal_True_1000_soap_0.01s_47963_N1000_nn_3.pth"
    elif nn_num == 4:
        test_file = "bow_mass_model_modmlp_Fb_iscausal_True_ichunk_13_anneal_True_1000_soap_0.01s_128474_N1000_nn_4.pth"
    elif nn_num == 5:
        test_file = "bow_mass_model_modmlp_Fb_iscausal_True_ichunk_22_anneal_True_1000_soap_0.01s_103605_N1000_nn_5.pth"
    print('')
else:
    test_file = f"bow_mass_model_modmlp_Fb_anneal_False_{fb_value}_{optimizer_type}_{T}s_1000000_nn_num{nn_num}.pth"

test_path = os.path.join(root_path, test_file)

trainer = Trainer(isvisual=isvisual, T=T, Fb=fb_value, pqmax=pqmax, isanneal=isanneal, optimzer_type=optimizer_type,
                  iscausal=iscausal, p_ic= p_ic, q_ic =q_ic, nn_num=nn_num, sigma=sigma)
trainer.test(test_path)
# # # trainer.test_hessian(test_path)
# print('')