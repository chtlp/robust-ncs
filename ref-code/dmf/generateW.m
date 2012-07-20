function W = generateW(X,k_min,k_max)
%% generate a weight matrix 
%%          W(i,j)=1, if node i selects node j as its neighbor;
%%          W(i,j)=0, otherwise.
%% k_min: the minimal number of neighbors selected by each node
%% k_max: the maximal number of neighbors selected by each node
%%        If not given, k_max=k_min, i.e., a constant number is used by all
%%        nodes.

n = size(X,1); 

if k_min<=0%use all data
    W = double(X>0);
    return
end

if ~exist('k_max') || (k_min==k_max)
    k = k_min*ones(n,1);
else
    k = k_min+floor((k_max-k_min)*rand(n,1));
end

W = zeros(n,n);
for i=1:n
    xi = X(i,:);
    ids = find(xi>0);
    tt = randperm(length(ids));
    W(i,ids(tt(1:k(i)))) = 1;
end

return
