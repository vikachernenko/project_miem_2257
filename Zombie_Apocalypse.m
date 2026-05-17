Nx = 25;
Ny = 25;

NL = 15;
NZ = 10;
pKill = 0.3;
maxSteps = 200;

rand("state", 1);

% 0 - пусто, 1 - человек, 2 - зомби
grid = zeros(Nx, Ny);

% случайное размещение людей
placed = 0;
while placed < NL
    x = randi(Nx);
    y = randi(Ny);
    if grid(x,y) == 0
        grid(x,y) = 1;
        placed = placed + 1;
    end
end

% случайное размещение зомби
placed = 0;
while placed < NZ
    x = randi(Nx);
    y = randi(Ny);
    if grid(x,y) == 0
        grid(x,y) = 2;
        placed = placed + 1;
    end
end

humansHistory = zeros(maxSteps + 1, 1);
zombiesHistory = zeros(maxSteps + 1, 1);
humansHistory(1) = sum(grid(:) == 1);
zombiesHistory(1) = sum(grid(:) == 2);

figure("Name", "Зомби апокалипсис");
for t = 1:maxSteps
    [rows, cols] = find(grid > 0);
    order = randperm(length(rows));
    for k = 1:length(order)
        i = rows(order(k));
        j = cols(order(k));
        if grid(i,j) == 0
            continue;
        end
        type = grid(i,j);
        direction = randi(4);
        ni = i;
        nj = j;
        if direction == 1
            ni = mod(i-2, Nx) + 1;
        elseif direction == 2
            ni = mod(i, Nx) + 1;
        elseif direction == 3
            nj = mod(j-2, Ny) + 1;
        else
            nj = mod(j, Ny) + 1;
        end
        if grid(ni,nj) == 0
            grid(ni,nj) = type;
            grid(i,j) = 0;
        end
    end

    newGrid = grid;
    [zx, zy] = find(grid == 2);
    for z = 1:length(zx)
        i = zx(z);
        j = zy(z);
        neighbors = [
            mod(i-2,Nx)+1, j;
            mod(i,Nx)+1, j;
            i, mod(j-2,Ny)+1;
            i, mod(j,Ny)+1
        ];
        for n = 1:4
            x = neighbors(n,1);
            y = neighbors(n,2);
            if grid(x,y) == 1
                if rand < pKill
                    newGrid(i,j) = 0;
                else
                    newGrid(x,y) = 2;
                end
            end
        end
    end
    grid = newGrid;

    % статистика
    humansHistory(t+1) = sum(grid(:) == 1);
    zombiesHistory(t+1) = sum(grid(:) == 2);
    if mod(t,5) == 0 || t == 1 || t == maxSteps
        subplot(1,2,1);
        imagesc(grid);
        axis equal tight;
        title(["Зомби-апокалипсис, шаг = " num2str(t)]);
        colormap([1 1 1; 0 0 1; 1 0 0]);
        colorbar;

        subplot(1,2,2);
        plot(0:t, humansHistory(1:t+1), "-o", 0:t, zombiesHistory(1:t+1), "-x");
        legend("Люди", "Зомби");
        xlabel("Шаг");
        ylabel("Количество");
        title("Динамика популяций");
        set(gca, "xgrid", "on", "ygrid", "on");
        drawnow;
    end

    % Кто победил
    if humansHistory(t+1) == 0 || zombiesHistory(t+1) == 0
        break;
    end
end

if humansHistory(t+1) == 0
    disp("Победили зомби");
elseif zombiesHistory(t+1) == 0
    disp("Победили люди");
else
    disp("Достигнут лимит шагов");
end

pause;
