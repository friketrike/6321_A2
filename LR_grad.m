function [ w, w_inits ] = LR_grad( X, y, w_inits, num_rand_inits )
%% Takes X, a matrix with all the input data, one instance per
% row, including a column of ones for the bias term; Y, the class
% labels in a column vector; W_INITS is an optional matrix of random 
% initial weight vectors where each vector is a column of the matrix;
% and optionally a number of random initial vectors which can be 
% generated inside the function if W_INITS is empty. Returns the 
% MxN matrix W of the weights found after GD for each vector, 
% and the ininitial random vectors (in case none were supplied).

    % default to 24 random starting points
    if nargin < 4 || isempty(num_rand_inits)
       num_rand_inits = 24; 
    end

    % if no random starting points are given, make some
    if nargin < 3 || isempty(w_inits)
        w_inits = rand(size(X,2), num_rand_inits); 
    else
        num_rand_inits = size(w_inits(2));
    end
    
    w = w_inits;
    for i = 1: num_rand_inits
        done = false;
        prev_epsilon = realmax * ones(length(y),1);
        iterations = 1;
        % find a good learning rate for each vector
        lr = find_alpha(X, y, w(:,i));
        % Don't allow GD to run after it's found something
        % nor for too long
        while ~done && (iterations < 2000)
            epsilon = y - (1./(1+exp(-X*w(:,i))));
            % if our prediction error is very small or it's not really
            % changing, we're done
            if (abs(sum(epsilon)) < .1) ...
                    || abs(sum(epsilon)-sum(prev_epsilon)) < 0.01
                done = true;
            else % keep going down
                w(:,i) = w(:,i) + lr * (X'*epsilon);
                prev_epsilon = epsilon;
            end
            iterations = iterations + 1;
        end
    end
end

