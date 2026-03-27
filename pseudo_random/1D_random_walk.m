clear; clc;

N_steps = [1, 2, 4, 8, 16, 32, 64, 128, 256];
num_walkers = 10000;
p = 0.5;
sigma_1d = zeros(size(N_steps));

figure;
for k = 1:length(N_steps)
    N = N_steps(k);
    final_pos = zeros(num_walkers, 1);

    for i = 1:num_walkers
        steps = 2 * (rand(1, N) < p) - 1;
        final_pos(i) = sum(steps);
    end

    sigma_1d(k) = std(final_pos);
    subplot(3,3,k);
    hist(final_pos, 30);
    title(sprintf('N = %d', N));
    xlabel('x'); ylabel('Frequency');
    xlim([-N N]);
end

figure;
loglog(N_steps, sigma_1d, 'o-', 'linewidth', 2);
hold on;
loglog(N_steps, sqrt(N_steps), '--', 'linewidth', 1.5);
xlabel('Number of steps N');
ylabel('\sigma');
legend('Experiment', '\surd N', 'location', 'northwest');
title('1D Random Walk: \sigma(N)');
grid on;

printf('Для 1D: σ ~ √N, так как каждый шаг независим -- дисперсия линейна.\n');
