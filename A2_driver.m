
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
% idx_test = [51, 29, 87, 93, 59, 1, 90, 23, 39, 96]; looks nice on the
% plots... bad doctoring of data, slap on the hand...
idxs = randperm(size(phi, 1));
idx_training = idxs(1:89);
idx_test = idxs(90:99);

phi_training = phi(idx_training, :);
y_training = y(idx_training);
phi_test = phi(idx_test, :);
y_test = y(idx_test);

lambdas = 0:0.1:24;
lambdas = lambdas .^ 3;
w = zeros(length(lambdas), size(phi,2));
for lambda = lambdas
  idx = lambda == lambdas;
  % Train the model
  w(idx, :) = pinv(phi_training' * phi_training ...
                + (lambda * eye(size(phi_training, 2)))) ...
                * (phi_training' * y_training);
  h_phi_training(:, idx) = phi_training * w(idx, :)';
  j_h_training(idx) = rms(h_phi_training(:, idx) - y_training);
  % Now compare to the test
  h_phi_test(:, idx) = phi_test * w(idx, :)';
  j_h_test(idx) = rms(h_phi_test(:, idx) - y_test);                          
end

% Now plot
figure(2); % RMS train / test vs lambda
subplot(2,2,1)
plot(lambdas(1:35), j_h_training(1:35), '.k')
title('RMS L2 \lambda < 40')
xlabel('\lambda')
ylabel('\epsilon_{L2} RMS')
hold on
plot(lambdas(1:35), j_h_test(1:35), '.g')
legend('training set', 'test set')
hold off

subplot(2,2,2)
plot(lambdas, j_h_training, '.k')
title('RMS L2 \lambda')
xlabel('\lambda')
ylabel('\epsilon_{L2} RMS')
hold on
plot(lambdas, j_h_test, '.g')
legend('training set', 'test set')
hold off

%figure(3); % Weights w_{L2} as a function of lambda
subplot(2,2,3:4)
plot(lambdas, w(:,1), '.k')
title('w_n{L2} as a function of \lambda')
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
                      -2*(phi_training'*y_training), ...
                      lambda*[1,1,1,0;1,1,-1,0;1,-1,1,0;1,-1,-1,0;...
                              -1,1,1,0;-1,1,-1,0;-1,-1,1,0;-1,-1,-1,0],... 
                      [1;1;1;1;1;1;1;1]);
  h_phi_training_quad(:, idx) = phi_training * w_quad(idx, :)';
  j_h_training_quad(idx) = rms(h_phi_training_quad(:, idx) - y_training);
  % Now compare to the test
  h_phi_test_quad(:, idx) = phi_test * w_quad(idx, :)';
  j_h_test_quad(idx) = rms(h_phi_test_quad(:, idx) - y_test);
end

% Now plot
figure(3); % RMS train / test vs lambda
subplot(2,1,1)
plot(lambdas(1:25), j_h_training_quad(1:25), '.k')
title('RMS error for L1 regularization')
xlabel('\lambda')
ylabel('\epsilon_{L1} RMS')
hold on
plot(lambdas(1:25), j_h_test_quad(1:25), '.g')
legend('training set', 'test set')
hold off
subplot(2,1,2)
plot(lambdas(1:50), j_h_training_quad(1:50), '.k')
title('RMS error for L1 regularization')
xlabel('\lambda')
ylabel('\epsilon_{L1} RMS')
hold on
plot(lambdas(1:50), j_h_test_quad(1:50), '.g')
legend('training set', 'test set')
hold off

figure(4); % Weights as a function of lambda
subplot(2,2,1)
title('w_{L1} as a function of \lambda')
plot(lambdas, w_quad(:,2), 'xg')
hold on
plot(lambdas, w_quad(:,3), '.r')
legend('w_2{L1}', 'w_3{L1}')
axis([0, 3, -0.05, 0.05])
hold off
subplot(2,2,2)
plot(lambdas(1:23), w_quad(1:23,1), '.k')
title('w_{L1} as a function of \lambda')
hold on
plot(lambdas(1:23), w_quad(1:23,2), 'xg')
plot(lambdas(1:23), w_quad(1:23,3), '.r')
plot(lambdas(1:23), w_quad(1:23,4), '.')
legend('w_1{L1}', 'w_2{L1}', 'w_3{L1}','w_4{L1}')
hold off
subplot(2,2,3:4)
plot(lambdas(1:35), w_quad(1:35,1), '.k')
title('Trend of all coefficients')
hold on
plot(lambdas(1:35), w_quad(1:35,2), 'xg')
plot(lambdas(1:35), w_quad(1:35,3), '.r')
plot(lambdas(1:35), w_quad(1:35,4), '.')
legend('w_1{L1}', 'w_2{L1}', 'w_3{L1}','w_4{L1}')
hold off



