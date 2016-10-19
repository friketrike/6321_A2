

% get data...
X = load('wpbcx.dat');
X = [X, ones(size(X,1),1)];
y = load('wpbcy.dat');

num_rand_inits = 24;

w = LR_grad(X, y, [], [], 24);

for i=1:8
    sum(y - (1./(1+exp(-X*w(:,i)))))
end




