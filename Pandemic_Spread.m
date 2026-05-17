Nx = 20;
Ny = 20;
N = Nx * Ny;

maxSteps = 100;
initialInfected = 3;
lambda = 0.3;
gamma = 0.1;
mu = 0.01;
betaRewire = 0.1;

rand("state", 3);

% 0 - здоровый, 1 - зараженный, 2 - выздоровевший, 3 - умерший
networkNames = {"Решетка", "Малый мир"};
allI = zeros(maxSteps + 1, 2);

figure("Name", "Распространение инфекции");

for mode = 1:2
    networkType = networkNames{mode};
    A = sparse(N, N);
    for x = 1:Nx
        for y = 1:Ny
            i = sub2ind([Nx, Ny], x, y);
            neighbors = [
                mod(x-2, Nx)+1, y;
                mod(x, Nx)+1, y;
                x, mod(y-2, Ny)+1;
                x, mod(y, Ny)+1
            ];
            for k = 1:4
                j = sub2ind([Nx, Ny], neighbors(k,1), neighbors(k,2));
                A(i,j) = 1;
                A(j,i) = 1;
            end
        end
    end
    if mode == 2
        [ii, jj] = find(triu(A, 1));
        for e = 1:length(ii)
            if rand < betaRewire
                i = ii(e);
                j = jj(e);
                A(i,j) = 0;
                A(j,i) = 0;
                newj = randi(N);

                while newj == i || A(i,newj) == 1
                    newj = randi(N);
                end
                A(i,newj) = 1;
                A(newj,i) = 1;
            end
        end
    end
    state = zeros(N, 1);
    infected0 = randperm(N, initialInfected);
    state(infected0) = 1;
    S = zeros(maxSteps + 1, 1);
    I = zeros(maxSteps + 1, 1);
    R = zeros(maxSteps + 1, 1);
    D = zeros(maxSteps + 1, 1);

    S(1) = sum(state == 0);
    I(1) = sum(state == 1);
    R(1) = sum(state == 2);
    D(1) = sum(state == 3);

    for t = 1:maxSteps
        infectedMask = (state == 1);
        susceptibleMask = (state == 0);

        infectedNeighbors = A * double(infectedMask);
        infectionProbability = 1 - (1 - lambda) .^ infectedNeighbors;
        newInfected = susceptibleMask & (rand(N,1) < infectionProbability);

        infectedIndexes = find(infectedMask);
        recoverMask = false(N,1);
        dieMask = false(N,1);
        for q = 1:length(infectedIndexes)
            person = infectedIndexes(q);
            if rand < mu
                dieMask(person) = true;
            elseif rand < gamma
                recoverMask(person) = true;
            end
        end
        state(newInfected) = 1;
        state(recoverMask) = 2;
        state(dieMask) = 3;

        S(t+1) = sum(state == 0);
        I(t+1) = sum(state == 1);
        R(t+1) = sum(state == 2);
        D(t+1) = sum(state == 3);

        if mode == 2 && (mod(t,5) == 0 || t == 1 || t == maxSteps)
            subplot(1,2,1);
            imagesc(reshape(state, Nx, Ny));
            axis equal tight;
            title(["Малый мир, шаг = " num2str(t)]);
            colorbar;

            subplot(1,2,2);
            plot(0:t, S(1:t+1), "-o", 0:t, I(1:t+1), "-x", 0:t, R(1:t+1), "-s", 0:t, D(1:t+1), "-d");
            legend("Здоровые", "Зараженные", "Выздоровевшие", "Умершие");
            xlabel("Шаг");
            ylabel("Количество людей");
            title("Динамика заражения");

            drawnow;
        end
        if I(t+1) == 0
            break;
        end
    end
    allI(1:length(I), mode) = I;
end

figure("Name", "Сравнение сетей");
plot(0:maxSteps, allI(:,1), "-o", 0:maxSteps, allI(:,2), "-x");
legend("Решетка", "Малый мир");
xlabel("Шаг");
ylabel("Количество зараженных");
title("Распространения инфекции");
grid on;

pause;
