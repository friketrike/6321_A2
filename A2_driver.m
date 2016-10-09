
phi = load('hw2x.dat');
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

phi_training = phi(idx_training);
y_training = y(idx_training);
phi_test = phi(idx_test);
y_test = y(idx_test);

lambdas = 0:0.3:10;
lambdas = lambdas .^ 2;
for lambda = lambdas
  idx = lambda == lambdas;
  % Train the model
  w(:, idx) = (phi_training' * phi_training + (lambda * ones(size(phi, 2)))) ...
                \ (phi_training' * y_training);
  h_phi_training(:, idx) = phi_training * w(:, idx);
  j_h_training(:, idx) = (sum((h_phi_training - y_training).^2) ...
                            / (2*size(phi_training,1)))^0.5;
  % Now compare to the test
  h_phi_training(:, idx) = phi_training * w(:, idx);
  j_h_training(:, idx) = (sum((h_phi_training - y_training).^2) ...
                            / (2*size(phi_training,1)))^0.5;                          
end

% Now plot
figure(2); % RMS train / test vs lambda
figure(3); % Weights as a function of lambda

