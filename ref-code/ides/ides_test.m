function ides_test()
load('recon.mat');

%let's use the AMP dataset as an example
N = length(AMP); %should get 110 here
L = 20; % 20 landmarks
dim = 10; % 10 dimension vectors

tmp = randperm(N); 
landmarks = tmp(1:L); % choose random L nodes as landmarks
hosts = tmp(L+1:N);  % the rest of them are orinary hosts

D_landmark = AMP(landmarks, landmarks);
D_host2landmark = AMP(hosts, landmarks);

%The default ides algorithm, using SVD
[out_l, in_l, out_h, in_h] = ides(D_landmark, D_host2landmark, dim);
plot_relative_error(out_h*in_h, AMP(hosts, hosts), 'r');

%The other ides algorithm, using NMF. Note that NMF results depend
%on the random initial values, so they are different in each run.
[out_l, in_l, out_h, in_h] = ides_NMF(D_landmark, D_host2landmark, dim);
plot_relative_error(out_h*in_h, AMP(hosts, hosts), 'g');
