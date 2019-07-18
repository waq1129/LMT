% Unit test script

clear

%% Single trial, no temporal basis
X{1} = [1 2 3];
Y{1} = [1];
zaso = encapsulateTrials(X, Y);
assert(zasoFxsum(zaso, @(x) sum(x)) == sum(X{1}));
assert(zasoFysum(zaso, @(x) sum(x)) == sum(Y{1}));
assert(zasoFx(zaso, @(x) sum(x)) == sum(X{1}));
assert(zasoFy(zaso, @(x) sum(x)) == sum(Y{1}));
[rsum, ragg] = zasoFarray(zaso, {@(x,y) x' * x, @(x,y) x' * y}, {@(x,y) sum(x)});
assert(all(all(rsum{1} == X{1}' * X{1})));
assert(all(all(rsum{2} == X{1}' * Y{1})));
assert(ragg{1} == sum(X{1}));
assert(zaso.sub2idx(1) == 1);

%% Two trials, still no temporal basis
X{2} = [-3 3 5];
Y{2} = [-1];

zaso = encapsulateTrials(X, Y);
assert(zasoFxsum(zaso, @(x) sum(x)) == sum(cellfun(@sum, X)));
assert(zasoFysum(zaso, @(x) sum(x)) == sum(cellfun(@sum, Y)));
assert(all(zasoFx(zaso, @(x) sum(x)) == cellfun(@sum, X)));
assert(all(zasoFy(zaso, @(x) sum(x)) == cellfun(@sum, Y)));
[rsum, ragg] = zasoFarray(zaso, {@(x,y) x' * x, @(x,y) x' * y}, {@(x,y) sum(x)});
assert(all(all(rsum{1} == X{1}' * X{1} + X{2}' * X{2})));
assert(all(all(rsum{2} == X{1}' * Y{1} + X{2}' * Y{2})));
assert(all(ragg{1} == cellfun(@sum, X)));
assert(zaso.sub2idx(1) == 1);
assert(zaso.sub2idx(2) == 2);

%% Single trial, with a single temporal basis
clear X Y
X{1} = [10];
Y{1} = [10];
tbX = @(x) sum(exp(-(ones(length(x),1)*(1:10)-x'*ones(1,10)).^2));
zaso = encapsulateTrials(X, Y, tbX);
assert(zasoFx(zaso, @(x) sum(x)) == sum(exp(-((1:10) - X{1}).^2)));
assert(zasoFxsum(zaso, @(x) sum(x)) == sum(exp(-((1:10) - X{1}).^2)));
assert(zasoFy(zaso, @(x) sum(x)) == sum(Y{1}));
assert(zaso.Nsub == 1);
assert(zaso.sub2idx(1) == 1);

%% Single trial, with two temporal bases
X{1} = [4];
Y{1} = [1];
tbX = @(x) [sum(exp(-(ones(length(x),1)*(1:10)-x'*ones(1,10)).^2),1), ...
	    sum(exp(-(ones(length(x),1)*(1:10)/2-x'*ones(1,10)).^2),1)];
zaso = encapsulateTrials(X, Y, tbX);
assert(zasoFx(zaso, @(x) sum(x)) == sum(tbX(X{1})))

%% Two trials with variable length
X{1} = [0 0 1 0 1]';
Y{1} = [0 0 1 0 1]';
X{2} = [-9 -10 2]';
Y{2} = [2.5 0 0]';
tbX = @(x) (exp(-(ones(size(x,1),1)*(1:10)-x*ones(1,10)).^2));
zaso = encapsulateTrials(X, Y, tbX);
assert(all(zaso.sub2idx(1) == 1:5));
assert(zaso.Nsub == 2);
assert(all(zaso.sub2idx(2) == 6:8));

disp('Passed! [$Id$]');
