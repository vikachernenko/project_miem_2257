%% Задача 1. Сложение двух матриц
clear; clc;

fprintf('Задача 1: сложение матриц \n');
tStart = tic;

M1 = load('M1.dat');
M2 = load('M2.dat');

if ~isequal(size(M1), size(M2))
    error('Размеры матриц не совпадают!');
end

M3 = M1 + M2;
save('M3.dat', 'M3', '-ascii');

tElapsed = toc(tStart);
fprintf('Сложение выполнено. Время: %.4f сек.\n', tElapsed);

%-------------------------------------------------------------------------------

%% Задача 2. Умножение двух матриц
fprintf('\n Задача 2: умножение матриц \n');

tStart = tic;
M1 = load('M1.dat');
M2 = load('M2.dat');

if size(M1,2) ~= size(M2,1)
    error('Количество столбцов M1 не равно количеству строк M2');
end

M3 = M1 * M2;
save('M3.dat', 'M3', '-ascii');

tElapsed = toc(tStart);
fprintf('Умножение выполнено. Время: %.4f сек.\n', tElapsed);

%-------------------------------------------------------------------------------

%% Задача 3. Инверсия трёхдиагональной матрицы (алгоритм Томаса)
fprintf('\n Задача 3: инверсия трёхдиагональной матрицы \n');

tStart = tic;
T_full = load('T.dat');
[n, m] = size(T_full);
if n ~= m
    error('Матрица должна быть квадратной!');
end

% Извлечение трёх диагоналей
a = zeros(n,1);  % нижняя
b = zeros(n,1);  % главная
c = zeros(n,1);  % верхняя
for i = 1:n
    b(i) = T_full(i,i);
    if i > 1
        a(i) = T_full(i,i-1);
    end
    if i < n
        c(i) = T_full(i,i+1);
    end
end

% Проверка диагонального преобладания (по условию задачи)
if any(abs(b) <= abs(a) + abs(c))
    warning('Условие диагонального преобладания может не выполняться!');
end

% Вычисление обратной матрицы алгоритмом Томаса
% Решаем n систем A * x_j = e_j, где e_j – j-й единичный вектор
% Решения x_j будут столбцами обратной матрицы

% выделяем массив для правых частей (все единичные векторы)
RHS = eye(n);
X = zeros(n);           % обратная матрица

% Для каждой правой части (столбца) решаем систему
for col = 1:n
    a_temp = a;
    b_temp = b;
    c_temp = c;
    d_temp = RHS(:, col);   % правая часть (j-й единичный вектор)

    % Прямой ход (метод Томаса)
    for i = 2:n
        w = a_temp(i) / b_temp(i-1);
        b_temp(i) = b_temp(i) - w * c_temp(i-1);
        d_temp(i) = d_temp(i) - w * d_temp(i-1);
    end

    % Обратный ход
    x = zeros(n,1);
    x(n) = d_temp(n) / b_temp(n);
    for i = n-1:-1:1
        x(i) = (d_temp(i) - c_temp(i) * x(i+1)) / b_temp(i);
    end

    X(:, col) = x;
end

save('T_inv.dat', 'X', '-ascii');

tElapsed = toc(tStart);
fprintf('Инверсия трёхдиагональной матрицы выполнена. Время: %.4f сек.\n', tElapsed);

% Вывод информации о невязке (для проверки)
I_calc = T_full * X;          % произведение матрицы на её обратную
err = norm(I_calc - eye(n), 'fro');
fprintf('Невязка ||T * T^{-1} - I||_F = %e\n', err);
