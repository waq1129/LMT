T = 50;
dx = 3;
X = sparse(zeros(T, dx));
X(1,1) = 1;
X(3,1) = 1;
X(5,2) = 1;
X(30,2) = 2;
X(45,3) = 1;
Y = randn(T, 1);

TB = 16;
bases(:, 1) = cos((1:TB)/15*pi);
bases(:, 2) = cos((1:TB)/30*pi);
M = size(bases, 2);

indices = false(dx, M);
indices(1, 1) = true;
indices(2, 1) = true;
indices(2, 2) = true;
indices(3, 2) = true;

BX = temporalBases_sparse(X, bases, indices)

XT{1} = X;
YT{1} = Y;
zaso = encapsulateTrials(XT, YT, @(x) temporalBases_sparse(x, bases, indices));

zasoFxsum(zaso, @(x) sum(x(:)))
zasoFx(zaso, @(x) x(:,4)')

figure(5754); clf; hold on
plot(zasoFx(zaso, @(x)full(x)')', 'o') % This should be same as BX
plot(BX, 'k-');
