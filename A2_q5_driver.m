

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
errors_gnb = zeros(num_folds,1);

% TODO perform cross-validation for 1 feature then all

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
    X_train = X(folds_idx([idxs_prev, idxs_next]), features);
    X_test = X(folds_idx(idxs_xcl), features);
    y_train = y(folds_idx([idxs_prev, idxs_next]));
    y_test = y(folds_idx(idxs_xcl));
    [w, w_init] = LR_grad(X_train, y_train, [], num_rand_inits);
    w_grad(fold, :, :) = w;
    for i=1:num_rand_inits
        errors_grad(fold, i) = sum(y_test ~= round(1./(1+exp(-X_test*w(:,i)))));
    end
    % remove the column of ones, GNB needs no bias as it's centered around
    % the mean...
    X_train(:,end) = [];
    X_test(:,end) = [];
    [theta, mu_1, mu_0, Sigma] = gnb_train(X_train,y_train);
    [log_odds, p1, p0] = gnb_predict(X_test, theta, mu_1, mu_0, Sigma);
    figure(1);
    plot(p1, 'g'); 
    hold on; 
    plot(p0, 'r'); 
    plot(y_test.*max(p0), 'k*')
    plot((log_odds > 0)*max(p0), 'o')
    title(['Class and GNB class prediction for fold ', num2str(fold)]);
    xlabel('instances of x');
    legend('P(y = 1|X)', 'P(y = 0|X)', 'y', 'y\^', 'location', 'east');
    hold off
    pause(0.5);
    errors_gnb(fold) = sum(((log_odds > 0) ~= y_test));
end

[min_err, idx] = min(abs(mean(errors_grad, 1)));
disp('For logistic regression:');
disp(sprintf(['the minimum error is %d and is given ',...
    'by random start vector number %d:'], min_err, idx));

err = abs(mean(errors_gnb));
disp('For Gaussian Naive Bayes:');
disp(sprintf('the error is %d', err));

% for i = 1:10
%     figure(2)
%     plot(w_grad(i,:,idx))
%     hold on
% end
% plot(mean(w_grad(:,:,idx),1), 'r--o')
% title('Resulting w coefficients over the 10 folds');
% hold off
% legend('w_{f_1}', 'w_{f_2}', 'w_{f_3}', 'w_{f_4}',...
%        'w_{f_5}', 'w_{f_6}', 'w_{f_7}', 'w_{f_8}',...
%        'w_{f_9}', 'w_{f_{10}}', 'mean');

