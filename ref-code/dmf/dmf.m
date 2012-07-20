function [U,V,err,mae,Uall,Vall] = dmf(X, W, params)
%% decentralized matrix factorization on one way delay matrix 
%% approximated by half of RTT matrix
%%
%% min||W.*(X-U*V')||^2.
%% X: RTT/2 matrix with negative values representing missing data
%% W: Weight matrix with 0 representing unmeasured and 1 measured
%%

if size(X,1)~=size(X,2)
    disp('X has to be a square');
    U = 0;
    V = 0;
    err = 0;
    mae = 0;
    Uall = 0;
    Vall = 0;
    return
end

if ~exist('params')
    params.lambda = 50;%regularization coeffienct of coordinate
    params.dimension = 3;%number of dimensions
    
    params.maxIters = 50;
    params.inRandomOrder = 1;
    params.doReport = 1;
    params.showUV = 0;
end

lambda = params.lambda;
d = params.dimension;

if isfield(params,'maxIters')
    maxIters = params.maxIters;
else
    maxIters = 50;
end
if isfield(params,'doReport')
    doReport = params.doReport;
else
    doReport = 1;
end
if isfield(params,'inRandomOrder')
    inRandomOrder = params.inRandomOrder;
else
    inRandomOrder = 1;
end
if isfield(params,'showUV')
    showUV = params.showUV;
else
    showUV = 0;
end

n = size(X,1);
epoch = n; 

%random initilization
U = rand(n,d);
V = rand(n,d);
Uall = zeros(n,d,maxIters);
Vall = zeros(n,d,maxIters);
Uall(:,:,1) = U;
Vall(:,:,1) = V;

err = zeros(maxIters,1);
mae = zeros(maxIters,1);

Xhat = U*V';
err(1) = stress(X,Xhat); 
mae(1) = medianAbsoluteError(X,Xhat); 

s = cputime;
for k = 2 : maxIters
    for l = 1 : epoch
        if inRandomOrder
            i = ceil(n*rand(1,1));%pick a node randomly
        else
            i = mod(l,n);
            if i==0, i=n; end;
        end

        ni = find(W(i,:)==1);

        A = V(ni,:);
        b = X(i,ni)';
        ui = inv(A'*A+lambda*eye(d))*A'*b;
        U(i,:) = ui';

        A = U(ni,:);
        b = X(ni,i);
        vi = inv(A'*A+lambda*eye(d))*A'*b;
        V(i,:) = vi';
    end
    
    Xhat = U*V';
    err(k) = stress(X,Xhat); 
    mae(k) = medianAbsoluteError(X,Xhat);
    Uall(:,:,k) = U;
    Vall(:,:,k) = V;

    diff = err(k-1)-err(k);
    % disp(sprintf('the improvment after the %dth iteration is %f',k,diff))
    %if diff<1e-5, break; end % exit iteration if little or no improvment
    
    if showUV & mod(k,1)==0
        figure(100), clf
        subplot(1,2,1), axis manual, plot(U(:,1),U(:,2),'r*')
        subplot(1,2,2), plot(V(:,1),V(:,2),'r*')
        drawnow 
    end
end

err = err(1:k);
mae = mae(1:k); 

if doReport
    disp('-------------decentralized matrix factorization-------------------');
    disp('----------------------------------------------------------------');
    disp('|         error        | number of iterations | running time(seconds) |');
    disp('----------------------------------------------------------------');
    disp(sprintf('|       %0.5g          |          %d          |          %f          |',err(k),k,cputime-s));
    disp('----------------------------------------------------------------');
end

if k>=maxIters
    disp('maximum number of iterations reached, may not converge.');
end

return
