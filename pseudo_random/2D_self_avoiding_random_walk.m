clear; clc;

N_steps = 1:2:13;          % нечётные шаги, чтобы избежать чётности
num_successful = 2000;
max_attempts = 5000;

sigma_sa = zeros(size(N_steps));
figure;

for k = 1:length(N_steps)
    N = N_steps(k);
    distances = zeros(num_successful, 1);
    successful = 0;

    while successful < num_successful
        path_x = 0;
        path_y = 0;
        offset = N+1;
        visited = zeros(2*offset+1, 2*offset+1);
        visited(offset+1, offset+1) = 1;

        valid = true;
        for step = 1:N
            dirs = [1 2 3 4];  % 1 - вверх, 2 - вниз, 3 - влево, 4 - вправо
            found = false;
            while ~isempty(dirs)
                idx = randi(length(dirs));
                d = dirs(idx);
                dx = 0; dy = 0;
                if d == 1, dy = 1; end
                if d == 2, dy = -1; end
                if d == 3, dx = -1; end
                if d == 4, dx = 1; end
                new_x = path_x(end) + dx;
                new_y = path_y(end) + dy;

                if abs(new_x) > offset || abs(new_y) > offset
                    new_offset = max(offset, max(abs(new_x), abs(new_y))) + 5;
                    new_visited = zeros(2*new_offset+1, 2*new_offset+1);

                    old_off = offset;
                    new_off = new_offset;
                    new_visited(new_off+1-old_off:end, new_off+1-old_off:end) = visited;
                    visited = new_visited;
                    offset = new_offset;
                end
                if visited(new_x+offset+1, new_y+offset+1) == 0
                    path_x = [path_x, new_x];
                    path_y = [path_y, new_y];
                    visited(new_x+offset+1, new_y+offset+1) = 1;
                    found = true;
                    break;
                else
                    % если занято, удаляем это направление
                    dirs(idx) = [];
                end
            end
            if ~found
                valid = false;
                break;
            end
        end

        if valid
            successful = successful + 1;
            distances(successful) = sqrt(path_x(end)^2 + path_y(end)^2);
        end

        if successful + (max_attempts * k) > 2e6
            warning('Не удалось набрать %d траекторий для N=%d. Остановлено.', num_successful, N);
            break;
        end
    end

    sigma_sa(k) = std(distances(1:successful));

    subplot(3, ceil(length(N_steps)/3), k);
    hist(distances(1:successful), 30);
    title(sprintf('N = %d', N));
    xlabel('r'); ylabel('Frequency');
    xlim([0 N]);
end

figure;
loglog(N_steps, sigma_sa, 'o-', 'linewidth', 2);
hold on;
nu = 0.75;
loglog(N_steps, N_steps.^nu, '--', 'linewidth', 1.5);
xlabel('Number of steps N');
ylabel('\sigma');
legend('Experiment', sprintf('N^{%.2f}', nu), 'location', 'northwest');
title('2D Self-Avoiding Walk: \sigma(N)');
grid on;

printf('Для 2D самоизбегающего блуждания: σ ~ N^{ν}, ν ≈ 0.75 в 2D.\n');
