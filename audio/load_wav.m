clear

load("f_1000/f_1000_deep.mat");
u = 2 * (u - min(u)) / (max(u) - min(u)) - 1;
audiowrite('f_1000_deep.wav', u, fs);

load("f_1000/f_1000_fd.mat");
u = 2 * (u - min(u)) / (max(u) - min(u)) - 1;
audiowrite('f_1000_fd.wav', u, fs);

load("f_1000/f_1000_pinn.mat");
u = 2 * (u - min(u)) / (max(u) - min(u)) - 1;
audiowrite('f_1000_pinn.wav', u, fs);

