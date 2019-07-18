% Test script for generating/inspecting raised-cosine basis

basprs.nh = 10;  % number of basis vectors
basprs.hdt = .01; % time bin size (relative to 1 frame of stimulus)
basprs.endpoints = [0 10]; % peak of 1st and last basis vector
basprs.b = .25; % controls log vs. linear spacing (higher => more linear)

[ttvec, bborth,bb] = makeRaisedCosBasis(basprs);

subplot(211);
plot(ttvec, bb);
subplot(212);
semilogx(ttvec,bb);