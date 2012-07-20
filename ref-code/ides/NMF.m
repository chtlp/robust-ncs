function [W, H] = NMF(V, r)
%% result should be V = W H
%% r is the lower dimension
%% V are column vectors

    [n m]=size(V); % V contains your data in its column vectors
	
    M = V >= 0;
    Vpos = V .* (V >= 0);
    
    maxiter=200; % choose the maximum number of iterations
    W=rand(n,r); % randomly initialize basis
    H=rand(r,m); % randomly initialize encodings
    for iter=1:maxiter
        H = H.* ( ( W' * Vpos )./( W' * (W*H .* M) ));
        W = W.* ( ( Vpos * H' )./( (W*H .* M) * H') );
    end
    return

