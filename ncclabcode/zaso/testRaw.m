%%
N = 20000;
X = randn(N, 4);
wTrue = [3 2 1 -1]';
Y = X * wTrue + 0.1 * randn(N, 1);

zaso = encapsulateRaw(X, Y);

NTest = 1000;
XTest = randn(NTest, 4);
YTest = XTest * wTrue + 0.1 * randn(NTest, 1);

zasoTest = encapsulateRaw(XTest, YTest);

%% Obtaining moments of data
EX = zasoFxsum(zaso, @(x) sum(x)) / zaso.N;
EY = zasoFysum(zaso, @(y) sum(y)) / zaso.N;
EXX = zasoFxsum(zaso, @(x) x' * x) / zaso.N;
EXY = zasoFxysum(zaso, @(x,y) x' * y) / zaso.N; % for 1-D y

%% Basic linear regression example
XX = zasoFxsum(zaso, @(x) x' * x);
XY = zasoFxysum(zaso, @(x,y) x' * y);

w = XX \ XY;

mseTest = zasoFxysum(zasoTest, @(x,y) sum((x * w - y).^2)) / zasoTest.N
prediction = zasoFx(zasoTest, @(x) x * w);

%% Compute quantities in batch at once (faster than calling them separately)
[rsum, ragg] = zasoFarray(zaso, ...
		{@(x,y) sum(x), @(x,y) sum(y)}, ...
		{@(x,y) x*w, @(x,y) (x*w).^2});

%% Quadratic basis (1, x, x^2) regression example
% no cross-terms here
fX = @(X) [ones(size(X, 1), 1), X, X.^2];
Y = 3 + X * wTrue + 0.2 * X.^2 * wTrue + 0.1 * randn(N, 1);
zaso = encapsulateRaw(X, Y, fX);
YTest = 3 + XTest * wTrue + 0.2 * XTest.^2 * wTrue + 0.1 * randn(NTest, 1);
zasoTest = encapsulateRaw(XTest, YTest, fX);

[rsum, ragg] = zasoFarray(zaso, {@(x,y) x' * x, @(x,y) x' * y}, {});
XX = rsum{1};
XY = rsum{2};
w = XX \ XY;

mseTest = zasoFxysum(zasoTest, @(x,y) sum((x * w - y).^2)) / zasoTest.N
prediction = zasoFx(zasoTest, @(x) x * w);

%% subindexing test
zasoFx(zaso, @(x) sum(x'), 1:10)
zasoFx(zaso, @(x) sum(x'), 1)
zasoFxsum(zaso, @(x) sum(x'), 1)

%% Bulk-mode
isBulkProcess = true;
zasoB = encapsulateRaw(X, Y, fX, [], isBulkProcess);
assert(any(any(zasoFxsum(zaso, @(x) x' * x) == zasoFxsum(zasoB, @(x) x' * x))));
