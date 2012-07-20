function rerr = relative_error(estimate, real)
    rerr=abs(estimate-real)./(min(estimate,real)+1);
    %rerr=abs(log((estimate+0.01)./(real+0.01))./log(2));
    %rerr=abs(estimate-real)./(real+0.1);
    %mask  = (real>0) | (abs(estimate-real)>10);
    mask  = (real>0);
    rerr= rerr.*mask; 
    rerr = rerr(:);