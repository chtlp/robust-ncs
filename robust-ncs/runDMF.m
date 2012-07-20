function P = runDMF( M, W, dim, round )
addpath ../ref-code/dmf

params.lambda = 50;%regularization coeffienct of coordinate
params.dimension = dim;%number of dimensions
params.maxIters = round;
params.inRandomOrder = 1;
params.doReport = 0;
params.showUV = 0;

[U,V,err,mae,Uall,Vall] = dmf(M, W, params);

P= U*V';

rmpath ../ref-code/dmf
end

