

PoWER_uncor_learned
PoWER_uncor_constant
PoWER_cor_bk_constant
PoWER_cor_bk_learned
close all
load('uncor_constant.txt');
load('uncor_learned.txt');
load('cor_bk_constant.txt');
load('cor_bk_learned.txt');

figure,
plot(uncor_constant)
hold on
plot(uncor_learned)
hold on
plot(cor_bk_constant)
hold on
plot(cor_bk_learned)
xlim([0, 1200])
title('Comparison of PoWER learning curves with different exploration rates')
ylabel('Normalized return')
xlabel('Number of rollouts')
legend('Uncorrelated constant','Uncorrelated updating','Correlated constant','Correlated updating')
set(gca,'FontSize',12)
set(gca,'FontWeight','bold')
hold off
