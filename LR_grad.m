function [ w, w_inits ] = LR_grad( X, y, w_inits, num_rand_inits )
%LR_GRAD Summary of this function goes here
%   Detailed explanation goes here

    % default to 4 random starting points
    if nargin < 4 || isempty(num_rand_inits)
       num_rand_inits = 4; 
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
        lr = find_alpha(X, y, w(:,i));
        while ~done && (iterations < 20000)
            epsilon = y - (1./(1+exp(-X*w(:,i))));
            if (abs(sum(epsilon)) < .01) ...
                    || abs(sum(epsilon)-sum(prev_epsilon)) < 0.001
                done = true;
            else
                w(:,i) = w(:,i) + lr * (X'*epsilon);
                prev_epsilon = epsilon;
            end
            iterations = iterations + 1;
        end
    end
end

