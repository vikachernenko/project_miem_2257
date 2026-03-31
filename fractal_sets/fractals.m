function run_fractals()
    width  = 4000;
    height = 4000;
    R = 2.0;
    maxIter = 500;
    colormap_name = 'hot';

    xMin = -2.5;  xMax = 1.5;
    yMin = -1.5;  yMax = 1.5;

    julia_c_values = [-0.7 + 0.27i, 0.3 + 0.6i, -0.8 + 0.156i, 0.365 - 0.37i];

    burning_xMin = -2.0;  burning_xMax = 1.0;
    burning_yMin = -1.5;  burning_yMax = 1.5;

    fprintf('Визуализация множества Мандельброта(%d x %d, maxIter = %d)...\n', width, height, maxIter);
    tic;
    mandelbrot_img = renderMandelbrot(width, height, xMin, xMax, yMin, yMax, maxIter, R, colormap_name);
    mandelbrot_time = toc;
    fprintf('Множество Мандельброта посчитано за %.2f seconds.\n', mandelbrot_time);
    imwrite(mandelbrot_img, 'mandelbrot.png');
    fprintf('Сохранено: mandelbrot.png\n\n');

    for idx = 1:length(julia_c_values)
        c = julia_c_values(idx);
        fprintf('Визуализация множества Жюлиа для c = %s (%d x %d, maxIter = %d)...\n', num2str(c), width, height, maxIter);
        tic;
        julia_img = renderJulia(width, height, xMin, xMax, yMin, yMax, maxIter, R, c, colormap_name);
        julia_time = toc;
        fprintf('Множество Жюлиа посчитано за %.2f seconds.\n', julia_time);
        filename = sprintf('julia_%s.png', strrep(num2str(c), ' ', ''));
        imwrite(julia_img, filename);
        fprintf('Сохранено: %s\n\n', filename);
    end

    fprintf('Визуализация горящего корабля (%d x %d, maxIter = %d)...\n', width, height, maxIter);
    tic;
    burning_img = renderBurningShip(width, height, burning_xMin, burning_xMax, burning_yMin, burning_yMax, maxIter, R, colormap_name);
    burning_time = toc;
    fprintf('Фрактал посчитан за %.2f seconds.\n', burning_time);
    imwrite(burning_img, 'burning_ship.png');
    fprintf('Сохранено: burning_ship.png\n');
end


function img = renderMandelbrot(width, height, xMin, xMax, yMin, yMax, maxIter, R, cmap)
    x = linspace(xMin, xMax, width);
    y = linspace(yMin, yMax, height);
    [X, Y] = meshgrid(x, y);
    C = X + 1i*Y;

    Z = zeros(size(C));
    iterCount = zeros(size(C));
    escaped = false(size(C));

    for n = 1:maxIter
        mask = ~escaped;
        Z(mask) = Z(mask).^2 + C(mask);
        newEscaped = mask & (abs(Z) > R);
        iterCount(newEscaped) = n;
        escaped = escaped | newEscaped;
        if all(escaped(:))
            break;
        end
    end
    iterCount(~escaped) = maxIter;

    mu = double(iterCount);
    escapedMask = (iterCount < maxIter);
    logz = log(abs(Z(escapedMask)));
    loglogz = log(logz);
    mu(escapedMask) = mu(escapedMask) + 1 - loglogz / log(2);

    mu = min(max(mu, 0), maxIter);
    mu_norm = mu / maxIter;
    mu_norm = max(0, min(1, mu_norm));

    idx = gray2ind(mu_norm, 256);
    img = ind2rgb(idx, colormap(cmap));
    img = uint8(img * 255);
end

function img = renderJulia(width, height, xMin, xMax, yMin, yMax, maxIter, R, c, cmap)
    x = linspace(xMin, xMax, width);
    y = linspace(yMin, yMax, height);
    [X, Y] = meshgrid(x, y);
    Z = X + 1i*Y;

    iterCount = zeros(size(Z));
    escaped = false(size(Z));

    for n = 1:maxIter
        mask = ~escaped;
        Z(mask) = Z(mask).^2 + c;
        newEscaped = mask & (abs(Z) > R);
        iterCount(newEscaped) = n;
        escaped = escaped | newEscaped;
        if all(escaped(:))
            break;
        end
    end
    iterCount(~escaped) = maxIter;

    mu = double(iterCount);
    escapedMask = (iterCount < maxIter);
    logz = log(abs(Z(escapedMask)));
    loglogz = log(logz);
    mu(escapedMask) = mu(escapedMask) + 1 - loglogz / log(2);
    mu = min(max(mu, 0), maxIter);
    mu_norm = mu / maxIter;
    mu_norm = max(0, min(1, mu_norm));

    idx = gray2ind(mu_norm, 256);
    img = ind2rgb(idx, colormap(cmap));
    img = uint8(img * 255);
end

function img = renderBurningShip(width, height, xMin, xMax, yMin, yMax, maxIter, R, cmap)
    x = linspace(xMin, xMax, width);
    y = linspace(yMin, yMax, height);
    [X, Y] = meshgrid(x, y);
    C = X + 1i*Y;

    Z = zeros(size(C));
    iterCount = zeros(size(C));
    escaped = false(size(C));

    for n = 1:maxIter
        mask = ~escaped;
        Z(mask) = (abs(real(Z(mask))) + 1i*abs(imag(Z(mask)))).^2 + C(mask);
        newEscaped = mask & (abs(Z) > R);
        iterCount(newEscaped) = n;
        escaped = escaped | newEscaped;
        if all(escaped(:))
            break;
        end
    end
    iterCount(~escaped) = maxIter;

    mu = double(iterCount);
    escapedMask = (iterCount < maxIter);
    logz = log(abs(Z(escapedMask)));
    loglogz = log(logz);
    mu(escapedMask) = mu(escapedMask) + 1 - loglogz / log(2);
    mu = min(max(mu, 0), maxIter);
    mu_norm = mu / maxIter;
    mu_norm = max(0, min(1, mu_norm));

    idx = gray2ind(mu_norm, 256);
    img = ind2rgb(idx, colormap(cmap));
    img = uint8(img * 255);
end
