

% get data...
X = load('wpbcx.dat');
X = [X, ones(size(X,1),1)];
y = load('wpbcy.dat');

w = LR_grad(X, y);




