% Poisson regression on a temporal basis

rand('seed', 20120911+3); randn('seed', 20120911+3);

addpath('..', '../regression_tools', '../nlfuns');

%% Make true event kernels
% distinct event types each induce changes in the firing rate
nEvents = 3;
eventKernelDuration = 200;
eventKernels = zeros(eventKernelDuration, nEvents);
bias = -2.8;
for k = 1:nEvents
    eventKernels(:, k) = (15 + 8 * rand) * normpdf(1:eventKernelDuration, 20 * rand, 30 * rand);
end
temp = eventKernels(:, 1);
eventKernels(:, 1) = eventKernels(:, 3);
eventKernels(:, 3) = temp;
figure(391); clf; plot(eventKernels); title('true kernels'); xlabel('time');

spikeHistoryKernel = exp([-7 -5 -3 -1.5 -0.3 0 0 0.1 0.2 0.06 0.02 0.01 0.01 0 0]');
useSpikeHistory = true; % for estimation

%% generate random events
nTrials = 1000;
T = 1200; % trial duration
X = cell(nTrials, 1); % store the spike trains
for kTrial = 1:nTrials
    X{kTrial} = sparse(ceil(rand(nEvents,1) * (T - 100)), 1:nEvents, 1, T + length(spikeHistoryKernel), nEvents);
end

%% Simulate exponential Poisson neuron with spike history kernel
Y = cell(nTrials, 1);
y = zeros(T, 1);
for kTrial = 1:nTrials
    BX = temporalBases_sparse(X{kTrial}, eventKernels, diag(true(nEvents, 1)));
    lambda = exp(sum(BX, 2) + bias);
    %clf; hold all; plot(lambda);
    for t = 1:T
	y(t) = poissrnd(lambda(t));
	if y(t) >= 1
	    y(t) = 1;
%	    lambda(t+1:t+length(spikeHistoryKernel)) = ...
%		max(0, lambda(t+1:t+length(spikeHistoryKernel)) ...
%		    + spikeHistoryKernel);
	    lambda(t+1:t+length(spikeHistoryKernel)) = ...
		lambda(t+1:t+length(spikeHistoryKernel)) .* spikeHistoryKernel;
	end
    end
    Y{kTrial} = y(1:T);
    %plot(lambda); return;
end

for kTrial = 1:nTrials
    X{kTrial} = X{kTrial}(1:T,:); % remove excess time bins
    if useSpikeHistory
    X{kTrial}(2:T,end+1) = Y{kTrial}(1:end-1); % add spike train itself as input
    end
end
if useSpikeHistory
    nEvents = nEvents + 1;
end

fprintf('Mean (Max) # of spikes per bin [%g (%d)]\n', mean(cellfun(@sum, Y)) / T, max(cellfun(@max, Y)))

tic;

%% Form a nice basis set
basprs.nh = 10;  % number of basis vectors
basprs.hdt = 1; % time bin size (relative to 1 frame of stimulus)
basprs.endpoints = [0 80]; % peak of 1st and last basis vector
basprs.b = .5; % controls log vs. linear spacing (higher => more linear)

[ttvec, bborth, bb] = makeRaisedCosBasis(basprs);
bb = bborth;
figure(1242); plot(ttvec, bb);

% all input uses all bases
basisIndices = true(nEvents, basprs.nh);
addDC = true;
zaso = encapsulateTrials(X, Y, @(x) temporalBases_sparse(x, bb, basisIndices, addDC));

%% Least-squares regression
[rsum, ragg] = zasoFarray(zaso, {@(x,y) x' * x, @(x,y) x' * y}, {});
XX = rsum{1};
XY = rsum{2};
wmap0 = XX \ XY;

%% reconstruct event kernels and plot them
figure(88599); clf;
estimatedEventKernel = temporalBases_combine(wmap0, bb, basisIndices, addDC);
for kEvent = 1:(nEvents-useSpikeHistory)
    subplot(nEvents, 1, kEvent); hold on;
    plot(eventKernels(:, kEvent), 'k');
    plot(estimatedEventKernel(:, kEvent), 'r');
end
if useSpikeHistory
kEvent = nEvents;
subplot(nEvents, 1, kEvent); hold on;
plot(log(spikeHistoryKernel), 'k');
plot(estimatedEventKernel(:, kEvent), 'r');
end

%beep; return;

%% Poisson
rhovals = 10.^(0:6)'; % grid over prior precision (hyperparameter)
rdgInds = 1:zaso.dimx; % indices to apply ridge parameter to
rhoNull = .01;  % prior precision for other variables
fnlin = @expfun;
[wRidge,rhoHat,SDerrbars,Hess] = autoRegress_PoissonRidge(zaso,[],fnlin,rdgInds,rhoNull,rhovals,wmap0);

%% reconstruct the kernel per event by combining the basis with the weights
estimatedEventKernel = temporalBasis_combine(wRidge, bb, basisIndices, addDC);
credibleInterval = 2 * sqrt(temporalBasis_combine(SDerrbars.^2, bb, basisIndices, addDC));

toc;

for kEvent = 1:(nEvents-1)
    cla; hold all;
    plot(eventKernels(:, kEvent), 'k');
    errorbar(estimatedEventKernel(:, kEvent), credibleInterval(:, kEvent), 'r');
    beep; pause
end
cla; hold all; kEvent = nEvents;
plot(log(spikeHistoryKernel), 'k');
plot(estimatedEventKernel(:, kEvent), 'r');
beep; pause
