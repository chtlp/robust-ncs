%% load the data and create neigbhor matrix
M = load('../data-set/p2psim_matrix.txt');
W = select_neigbhors(M, 32);

%% run robust matrix factorization
dim = 6;
round = 100;
lambda = 7;
U0 = rand(1740, 6);
V0 = rand(1740, 6);

P1 = runRobustMF(M, W, dim, round, lambda, U0, V0);

%% run robust Vivaldi (with height)
dim = 2;
round = 40;
lambda = 7;

% the last dim of P is the height vector
P2 = runRobustVivaldi(M, W, dim, round, lambda);