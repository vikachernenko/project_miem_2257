clear; clc;

N_steps = [1, 2, 4, 8, 16, 32, 64, 128, 256];
num_walkers = 10000;
p = 0.5;

sigma_2d = zeros(size(N_steps));

figure;
for k = 1:length(N_steps)
    N = N_steps(k);
    final_pos = zeros(num_walkers, 2);

    for i = 1:num_walkers
        steps_x = 2 * (rand(1, N) < p) - 1;
        steps_y = 2 * (rand(1, N) < p) - 1;
        final_pos(i,1) = sum(steps_x);
        final_pos(i,2) = sum(steps_y);
    end
    distances = sqrt(final_pos(:,1).^2 + final_pos(:,2).^2);
    sigma_2d(k) = std(distances);

    subplot(3,3,k);
    hist(distances, 30);
    title(sprintf('N = %d', N));
    xlabel('r'); ylabel('Frequency');
    xlim([0 N*sqrt(2)]);
end

figure;
loglog(N_steps, sigma_2d, 'o-', 'linewidth', 2);
hold on;
loglog(N_steps, sqrt(N_steps), '--', 'linewidth', 1.5);
xlabel('Number of steps N');
ylabel('\sigma');
legend('Experiment', '\surd N', 'location', 'northwest');
title('2D Random Walk: \sigma(N)');
grid on;

printf('Для 2D: σ ~ √N, так как движения по осям независимы, и r² = x²+y² имеет дисперсию, растущую линейно с N.\n');
