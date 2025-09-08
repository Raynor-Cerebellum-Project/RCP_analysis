
function [SE_out] = sd_err(Y,X,Yhat,n_imp)

n = max(size(Y));
k = sum(n_imp);

y = Y - mean(Y);
yhat = Yhat - mean(Y);
R2 = sum(yhat.^2) ./ sum(y.^2);
sy = sqrt( ( (1-R2).*sum(y.^2) ) ./ (n - k - 1) );

X1 = X;
z = find(n_imp == 0);
X1(:,z) = [];
X1 = [X1 ones(size(X1(:,1)))];
c = inv(X1'*X1);

V = []; SE = [];

%Get bias first
for i = 1:k+1,
   V(i)  = (c(i,i)).*sy.^2;
   SE(i) = sqrt(c(i,i)).*sy;
end

%Get the other parameters
z = find(n_imp == 1);
SE_out = [SE(i) 0 0 0 0];
for j = 1:sum(n_imp)
   SE_out(z(j)+1) = SE(j);
end