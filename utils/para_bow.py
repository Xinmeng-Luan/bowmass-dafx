'''
parameters: violin A4

In all cases the bow velocity started from zero and reached a maximum value through a linear ramp, was kept constant for
a while and then decreased linearly until zero.
The stopping of vibration was implemented by keeping the bowing force constant (and not null) after the bow velocity
reached zero. The Free case, on the other hand, was obtained by setting the force to zero as well. The amplitude
envelope of the samples reflects the forcing type: in the "Stop" case, the string is put into vibration, and has a fast
decay after the bow is stopped. In the "Free" case, the vibration continues after the bow is stopped, and the decay is slower.

no damping:
utt = c^2 uxx - K^2 uxxxx - Fb delta(x-xb) phi(eta)
'''

import numpy as np

seed = 9876
np.random.seed(seed)

# String Length & Fretting Position
baseLength = 0.32  #String base length
frettingPos = 1
L = baseLength*frettingPos

freq = 440 #[Hz]
radius = 3e-4 # string cross-section radius
rho = 2.5465e3 # string material density
T0 = 57.10 # applied tension
E = 19.5e9 # Young's modulus

excitPos = 0.833 # bowing point
# outPos1 = 0.33 * L
# outPos2 = 0.77 * L


A = np.pi* (radius**2)
rA = rho*A
Inertia = (np.pi* (radius**4))/ 4
K = np.sqrt(E*Inertia/(rA* (L**4)))
c = np.sqrt(T0/rA)


#---------------------------------- time, space points
T_tot = 5 #[s]
T= 1/freq #[s] TODO:test: periodic
# PDE loss
Npde = 8000 # ori:5000
# BC loss :x,[0,T]
Nbc = 1000
# ti_bc = np.linspace(0, T, num=Nbc)
# IC loss: [0,L],t
Nic = 1000
# xi_ic = np.linspace(0,L, num=Nic)
# bow force: excitPos,[0,T]
Nbow = 1000
# ti_bow = np.linspace(0, T, num=Nbow)

# bc0, bcL, ic, bow, pde
index_bc0 = range(0, Nbc)
index_bcL = range(Nbc, Nbc+Nbc)
index_ic0 = range(Nbc+Nbc, Nbc+Nbc+Nic)
index_icT = range(Nbc+Nbc+Nic, Nbc+Nbc+Nic+Nic)
index_bow = range(Nbc+Nbc+Nic+Nic, Nbc+Nbc+Nic+Nic+Nbow)
index_pde = range(Nbc+Nbc+Nic+Nic+Nbow, Nbc+Nbc+Nic+Nic+Nbow+Npde)

index_pde_bow = range(Nbc+Nbc+Nic+Nic, Nbc+Nbc+Nic+Nic+Nbow+Npde)

#---------------------------------- force
#### Fb
# - Fb delta(x-x_B) phi(eta)
# # free
# startFb = 30
# maxFb = 30
# endFb = 0

# not free
maxFb = 5

# Linear ramp for bow fractional force
frac = 3  # Number of sections in which the whole bow speed vector will be divided
# directly interp
# y = 0.06x + 0.1, x:[0,5/3]
# y = 0.2, x: [5/3, 10/3]
# y = -0.12x + 0.6, x: [10/3, 5]

# for t:[0,5e-3]
# maxVb = 0.06*T + 0.1
# startVb = 0.2
# vb = np.linspace(startVb, maxVb, Nbow)

Fb = maxFb

#### fractional: phi(eta): in loss function
fr_a = 100            # Bow free parameter
muD = 0.3          # Desvages friction parameter

# zeta1 = zetaTR*x
# eta = zeta1 - bowVel(i);
# if desvagesFriction
#     %Desvages friction
#     d = sqrt(2*a)*exp(-a*eta^2 + 0.5) + 2*muD*atan(eta/0.02)/pi/eta;
#     lambda = sqrt(2*a)*exp(-a*eta^2 + 0.5)*(1 - 2*a*eta^2) + 2*muD*50/pi/(2500*eta^2 + 1);
# else
#     %Bilbao friction
#     d = sqrt(2*a)*exp(-a*eta^2 + 0.5);
#     lambda = sqrt(2*a)*exp(-a*eta^2 + 0.5)*(1 - 2*a*eta^2);



# # damping: currently lossless
# # Desvages
# rhoAir = 1.225
# muAir = 1.619e-5
# d0 = -2 * rhoAir * muAir / (rho * r ^ 2)
# d1 = -2 * rhoAir * sqrt(2 * muAir) / (rho * r)
# d2 = -1 / 18000
# d3 = -0.003 * E * rho * pi ^ 2 * r ^ 6 / (4 * T0 ^ 2)
#
# sigma = d0 + d1 * sqrt(omega) + d2 * omega + d3 * omega ^ 3

#---------------nn
nn_width = 512 # in [128,512]
nn_depth = 6 # in [3,6]
norm_scale = 1.6e-4
lambda_pde = 3000 # utt
lambda_bc0 = 1.6e-12 # u
lambda_bcL = 1.6e-12 # u
# lambda_ic1 = 5e-6*5e-6/10000 # u
# lambda_ic2 = 0.014*0.014/10000 # ut
lambda_ic = 1.6e-12 # u
# lambda_pde = 1
# lambda_bc0 = 1
# lambda_bcL = 1
# lambda_ic1 = 1
# lambda_ic2 = 1

#---------------train_ok
lr = 1e-3
lr_gamma = 0.99999
epoch = 20000
epoch_save =500


loss_type = 'lossless'
bow_friction_type = 'Bilbao' # 'Desvages', 'Bilbao'
data_path = '/nas/home/xluan/thesis-xinmeng/BOW_PINN/bow_pinn/data/data2_period.pkl'
save_config_path = '/nas/home/xluan/thesis-xinmeng/BOW_PINN/bow_pinn/result/noloss/config.json'
save_result_path = '/nas/home/xluan/thesis-xinmeng/BOW_PINN/bow_pinn/result/noloss_no4order/'
net_arch = 'fcn'
ic = 'period'


