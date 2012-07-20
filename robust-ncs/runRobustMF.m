function P = runRobustMF(M, W, dim, round, lambda, U0, V0)

opt_sgd = struct('learn_rate', 0.005, 'huber_lambda', lambda, ...
             'L1Reg', 0.00, 'L2Reg', 0.0 , ...
             'display_freq', 50, 'save_model_freq', 50, 'Debug', false, ...
             'symm_factorization', false, 'BoldDrive', true);

W = W | W';

[U, V, models] = robust_mf_sgd(M, W, dim, round, opt_sgd, U0, V0);

P = U*V';
end