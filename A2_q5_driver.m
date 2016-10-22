

% get data...
X = load('wpbcx.dat');
X = [X, ones(size(X,1),1)];
y = load('wpbcy.dat');

num_folds = 10;

folds_info = cvpartition(length(y), 'kfold', num_folds);
folds_idx = randperm(length(y));

num_rand_inits = 24;

features = [1:33];
% features = [1,33];

w_grad = zeros(num_folds, length(features), num_rand_inits);
errors_grad = zeros(num_folds, num_rand_inits);

for fold = 1:num_folds
    disp(sprintf('Performing %d-fold CV, fold: %d', num_folds, fold));
    idxs_prev = 1:sum(folds_info.TestSize(1:(fold-1)));
    if ~isempty(idxs_prev)
        offset = idxs_prev(end);
    else
        offset = 0;
    end
    idxs_xcl = (1:folds_info.TestSize(fold))+offset;
    idx_after_skip = length(y)-(sum(folds_info.TestSize((fold+1):end))-1);
    idxs_next = idx_after_skip:length(y);
    X_train = X([idxs_prev, idxs_next], features);
    X_test = X(idxs_xcl, features);
    y_test = y(idxs_xcl);
    y_train = y([idxs_prev, idxs_next]);
    [w, w_init] = LR_grad(X_train, y_train, [], num_rand_inits);
    w_grad(fold, :, :) = w;
    for i=1:num_rand_inits
        errors_grad(fold, i) = sum(y_test - (1./(1+exp(-X_test*w(:,i)))));
    end
end

[min_err, idx] = min(abs(mean(errors_grad, 1)));
disp(sprintf(['the minimum error is %d and is given ',...
    'by random start vector number %d:'], min_err, idx));

for i = 1:10
plot(w_grad(i,:,21))
hold on
end
plot(mean(w_grad(:,:,21),1), 'r--o')
hold off
legend('1', '2', '3', '4', '5', '6', '7', '8', '9', '10', 'mean');

% TODO implement bayes?

% TODO check error of the whole set with the mean of the coefficients?

% TODO use cross-entropy for validation?

% TODO check the proximity of closest vectors to the one with min error

%%little snippet to check for solutions that fall close to each other
% for i = 1:num_rand_inits
%     for j = 1:num_rand_inits
%         norms(i,j) = norm(w(:,i)-w(:,j));
%     end
% end