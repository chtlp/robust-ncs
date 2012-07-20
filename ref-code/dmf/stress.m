function  s = stress(X,Xa)

ids = find(X>1e-5);

x = X(ids);
xa = Xa(ids);

d = x-xa;
s = sqrt((d'*d)/(x'*x));

return