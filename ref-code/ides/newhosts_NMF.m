function [new_w, new_h] =  newhosts_NMF(w,h, newd_out, newd_in, exist_lam)
% [new_w, new_h] =  newhosts_NMF(w,h, newd_out, newd_in, exist_lam)
% w: Nxd, h: dxN. The position vectors of all landmarks
% N: number of landmarks
% M: number of new hosts
% newd_out: Mxn_landmark out distance
% newd_in: n_landmarkxM in distance
% exist_lam: indexes of measured landmarks
% return: new_w: Mxd, new_h: dxM, such that new_w * h = newd
    [N,d] = size(w);
    [M,n_landmark] = size(newd_out);
    % nnls, mldivide
    %w = w(exist_lam,:);
    %h = h(:,exist_lam);
	
	exist_lam_mask = false(N, 1);
	exist_lam_mask(exist_lam) = true;
    for i=1:M
		valid = exist_lam_mask & (newd_out(i,:)' > 0);
		t = h(:,valid)' \ newd_out(i, valid)';
		new_w(i,:) = t';
		new_h(:,i) = w(valid,:) \ newd_in(valid, i);
		
        %t=h'\newd_out(i,exist_lam)';
        %new_w(i,:)=t';
        %new_h(:,i)=w\newd_in(exist_lam,i);
    end