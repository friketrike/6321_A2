

% get data...
X = load('wpbcx.dat');
X = [X, ones(size(X,1),1)];
y = load('wpbcy.dat');

num_rand_inits = 24;

[w, w_init] = LR_grad(X, y, [], [], num_rand_inits);

for i=1:num_rand_inits
    disp(sum(y - (1./(1+exp(-X*w(:,i))))))
end




