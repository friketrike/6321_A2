function [ w, w_inits ] = LR_grad( X, y, lr, w_inits, num_rand_inits )
%LR_GRAD Summary of this function goes here
%   Detailed explanation goes here
    if nargin < 3 || isempty(lr)
       lr = 1/3;
       % TODO, something smarter here to find a better learning rate
    end

    % default to 4 random starting points
    if nargin < 5 || isempty(num_rand_inits)
       num_rand_inits = 4; 
    end

    % if no random starting points are 
    if nargin < 4 || isempty(w_inits)
        w_inits = 10*rand(size(X,2), num_rand_inits); 
        %TODO, see how far apart these random points should be
        w_inits(:,1) = 0; % TODO remove...
    else
        num_rand_inits = size(w_inits(2));
    end
    
    w = w_inits;
    for i = 1: num_rand_inits
        done = false;
        prev_epsilon = realmax * ones(length(y),1);
        while ~done
            epsilon = y - (1./(1+exp(-X*w(:,i))));
            %i
            %mean(epsilon)
            w(:,i) = w(:,i) + lr * (X'*epsilon);
            % TODO, think about cutoff strategy, observe behaviour...
            if (abs(mean(prev_epsilon) - mean(epsilon)) < .0000001)
                done = true;
            else
                if abs(mean(epsilon)) < abs(mean(prev_epsilon))
                   lr = min([lr * 4/3,1])
                elseif abs(mean(epsilon)) > abs(mean(prev_epsilon))
                    lr = 2* lr / 3
                end
                prev_epsilon = epsilon;
            end
        end
    end
end

