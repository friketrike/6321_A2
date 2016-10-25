figure(4)
subplot(2,2,1)
i = 2; j = 5;
plot(X((y == 1),i), X((y == 1),j), 'g+')
hold on
plot(X((y == 0),i),X((y == 0),j), 'r.')
hold off
title(['feature ', num2str(i), ' against feature ', num2str(j)]);

subplot(2,2,2)
i = 11; j = 14;
plot(X((y == 1),i), X((y == 1),j), 'g+')
hold on
plot(X((y == 0),i),X((y == 0),j), 'r.')
hold off
title(['feature ', num2str(i), ' against feature ', num2str(j)]);

subplot(2,2,3)
i = 2;
plot(X(y==1, i), zeros(sum(y==1),1), 'g.')
hold on
plot(X(y==0, i), zeros(sum(y==0),1), 'r.')
exes = min(X(:,i))-1:0.1:max(X(:,i))+1;
plot(exes, normpdf(exes, mean(X(y == 1, i)), std(X(y == 1, i))), 'g')
plot(exes, normpdf(exes, mean(X(y == 0, i)), std(X(y == 0, i))), 'r')
title(['feature ', num2str(i), ' class distributions'])
hold off

subplot(2,2,4)
i = 11;
plot(X(y==1, i), zeros(sum(y==1),1), 'g.')
hold on
plot(X(y==0, i), zeros(sum(y==0),1), 'r.')
exes = min(X(:,i))-1:0.1:max(X(:,i))+1;
plot(exes, normpdf(exes, mean(X(y == 1, i)), std(X(y == 1, i))), 'g')
plot(exes, normpdf(exes, mean(X(y == 0, i)), std(X(y == 0, i))), 'r')
title(['feature ', num2str(i), ' class distributions'])
hold off
