Nx = 50;
Ny = 50;
maxSteps = 80;

forestDensity = 0.75;
omega = 0.15;
pBase = 0.002;
pAmplitude = 0.03;
rSpread = 0.35;
burnDuration = 3;
sExtinguish = 0.03;
plotEvery = 5;

rand("state", 2);

% Состояния: 0 - сгорело; 1 - лес; 2 - горит
grid = zeros(Nx, Ny);
grid(rand(Nx, Ny) < forestDensity) = 1;

% пожар в центре
grid(round(Nx/2), round(Ny/2)) = 2;

burnAge = zeros(Nx, Ny);
forestArea = zeros(maxSteps + 1, 1);
burningArea = zeros(maxSteps + 1, 1);
burntArea = zeros(maxSteps + 1, 1);

initialEmpty = sum(grid(:) == 0);
forestArea(1) = sum(grid(:) == 1);
burningArea(1) = sum(grid(:) == 2);
burntArea(1) = 0;

figure("Name", "Forest Fire Model");

for t = 1:maxSteps

    % вероятность случайного возгорания
    pIgnition = pBase + pAmplitude * sin(omega * t)^2;
    burningNeighbors = conv2(double(grid == 2), ones(3), "same") - double(grid == 2);

    nextGrid = grid;
    nextBurnAge = burnAge;
    forestMask = (grid == 1);
    igniteFromLightning = forestMask & (rand(Nx, Ny) < pIgnition);
    spreadProbability = 1 - (1 - rSpread) .^ burningNeighbors;
    igniteFromNeighbors = forestMask & (burningNeighbors > 0) & (rand(Nx, Ny) < spreadProbability);

    newFires = igniteFromLightning | igniteFromNeighbors;
    nextGrid(newFires) = 2;
    nextBurnAge(newFires) = 1;

    % клетки продолжают гореть
    burningMask = (grid == 2);
    nextBurnAge(burningMask) = nextBurnAge(burningMask) + 1;
    extinguishRandomly = burningMask & (rand(Nx, Ny) < sExtinguish);
    extinguishByAge = burningMask & (nextBurnAge >= burnDuration);

    stopBurning = extinguishRandomly | extinguishByAge;
    nextGrid(stopBurning) = 0;
    nextBurnAge(stopBurning) = 0;

    grid = nextGrid;
    burnAge = nextBurnAge;

    % статистика
    forestArea(t + 1) = sum(grid(:) == 1);
    burningArea(t + 1) = sum(grid(:) == 2);
    burntArea(t + 1) = sum(grid(:) == 0) - initialEmpty;
    if mod(t, plotEvery) == 0 || t == 1 || t == maxSteps
        clf;

        subplot(1,2,1);
        imagesc(grid);
        axis equal tight;
        title(["Горящий лес, шаг = " num2str(t)]);
        xlabel("y");
        ylabel("x");
        colormap([0.8 0.8 0.8; 0 0.6 0; 1 0 0]);
        colorbar;

        subplot(1,2,2);
        plot(0:t, forestArea(1:t+1), "-o", 0:t, burningArea(1:t+1), "-x", 0:t, burntArea(1:t+1), "-s");
        legend("Лес", "горение", "пламя");
        xlabel("Шаг");
        ylabel("Клетки");
        title(["ВерВозгарания = " num2str(pIgnition)]);
        drawnow;
    end
end

finalBurntFraction = burntArea(end) / (Nx * Ny);

disp(["Время горения = " num2str(finalBurntFraction)]);

pause;
