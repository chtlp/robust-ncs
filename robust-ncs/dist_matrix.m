function P = dist_matrix(C, use_height)
if nargin < 2
    use_height = false;
end
N = size(C, 1);

if use_height
    H = C(:,end);
    assert(all(H >= 0));
    C = C(:,1:end-1);
end

P = distmat(C);

if use_height
    P = P + repmat(H,1,N) + repmat(H',N,1);
end

end