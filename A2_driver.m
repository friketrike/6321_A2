
clear

phi = load('hw2x.dat');
phi = [phi, ones(size(phi,1),1)];
y = load('hw2y.dat');

figure(1)
plot(phi(:,1), y, '.');
hold on;
plot(phi(:,2), y, '.r');
plot(phi(:,3), y, '.k');
title('Confetti')
hold off

% Partition the data randomly
idxs = randperm(size(phi, 1));
idx_training = idxs(1:89);
idx_test = idxs(90:99);

phi_training = phi(idx_training, :);
y_training = y(idx_training);
phi_test = phi(idx_test, :);
y_test = y(idx_test);

lambdas = 0:0.1:15;
lambdas = lambdas .^ 3;
w = zeros(length(lambdas), size(phi,2));
for lambda = lambdas
  idx = lambda == lambdas;
  % Train the model
  w(idx, :) = pinv(phi_training' * phi_training ...
                + (lambda * eye(size(phi_training, 2)))) ...
                * (phi_training' * y_training);
  h_phi_training(:, idx) = phi_training * w(idx, :)';
  j_h_training(idx) = (sum((h_phi_training(:, idx) - y_training).^2) ...
                            / (2*size(phi_training,1))).^0.5;
  % Now compare to the test
  h_phi_test(:, idx) = phi_test * w(idx, :)';
  j_h_test(idx) = (sum((h_phi_test(:, idx) - y_test).^2) ...
                            / (2*size(phi_test,1))).^0.5;                          
end

% Now plot
figure(2); % RMS train / test vs lambda
plot(lambdas, j_h_training, '.k')
title('RMS error for L2 regularization')
xlabel('\lambda')
ylabel('\epsilon_{L2} RMS')
hold on
plot(lambdas, j_h_test, '.g')
legend('training set', 'test set')
hold off

figure(3); % Weights w_{L2} as a function of lambda
plot(lambdas, w(:,1), '.k')
title('w_{L2} as a function of \lambda')
xlabel('\lambda')
hold on
plot(lambdas, w(:,2), '.g')
plot(lambdas, w(:,3), '.r')
plot(lambdas, w(:,4), '.')
legend('w1_{L2}', 'w2_{L2}', 'w3_{L2}', 'w4_{L2}')
axis([0, 3500, -0.5, 0.17])
hold off

w_quad = zeros(length(lambdas), size(phi,2));
for lambda = lambdas
  idx = lambda == lambdas;
  w_quad(idx,:) = quadprog(2*(phi_training'*phi_training), ...
                      2*(y_training'*phi_training), ...
                      lambda*[1,1,1,0;1,1,-1,0;1,-1,1,0;1,-1,-1,0;...
                              -1,1,1,0;-1,1,-1,0;-1,-1,1,0;-1,-1,-1,0],... 
                      [1;1;1;1;1;1;1;1]);
  h_phi_training_quad(:, idx) = phi_training * w_quad(idx, :)';
  j_h_training_quad(idx) = (sum((h_phi_training_quad(:, idx) - y_training).^2) ...
                            / (2*size(phi_training,1))).^0.5;
  % Now compare to the test
  h_phi_test_quad(:, idx) = phi_test * w_quad(idx, :)';
  j_h_test_quad(idx) = (sum((h_phi_test(:, idx) - y_test).^2) ...
                            / (2*size(phi_test,1))).^0.5;
end

% Now plot
figure(4); % RMS train / test vs lambda
plot(lambdas, j_h_training_quad, '.k')
title('RMS error for L1 regularization')
xlabel('\lambda')
ylabel('\epsilon_{L1} RMS')
hold on
plot(lambdas, j_h_test_quad, '.g')
legend('training set', 'test set')
hold off

figure(5); % Weights as a function of lambda
plot(lambdas, w_quad(:,1), '.k')
title('w_{L1} as a function of \lambda')
hold on
plot(lambdas, w_quad(:,2), 'xg')
plot(lambdas, w_quad(:,3), '.r')
legend('w_1{L1}', 'w_2{L1}', 'w_3{L1}')
axis([0, 400, -0.04, 0.1])
hold off
