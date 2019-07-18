%% make figs showing 1D bump example (comparing LLE, PLDS, and P-GPLVM)
%
%  Run after running demo2_1DBump_jp.m

datasetname = 'simdatadir/simdata2_jp4.mat';  % name of dataset
load(datasetname);
xx = simdata.latentVariable;
yy = simdata.spikes;
ff = simdata.spikeRates;
nf = 1; % number of latent dimension

% % Generate tuning curves ff
% xgrid = gen_grid([-4 4;-4 4],200,nf); % x grid (for plotting purposes)
% fffull = ffun([xx;xgrid]); % generate ff for xx and for grid
% ff = fffull(1:nt,:); % sampled ff values
% ffgrid = fffull(nt+1:end,:); % for plotting purposes only

lw = 2;
plot(xx, ff*10, 'linewidth', 2); set(gca,'xlim', 3.5*[-1 1]);
%title('tuning curves');
xlabel('latent variable');
ylabel('firing rate (sp/s)');

set(gcf,'color', 'w');
%print -dpdf 

%% plot true latent 


clf;
tt = 1:300;
iiplot = 650+tt;
plot((tt)*.1, xx(iiplot), 'k', 'linewidth', lw); 
ylm = [-3 3];
set(gca,'ylim', ylm);
xlabel('time (s)');
ylabel('latent variable');

%% plot firing rates

%imagesc(tt*.1,1:20,log(10*ff(iiplot,:)')); axis xy;
imagesc(tt*.1,1:20,10*ff(iiplot,:)'); axis xy;
%set(gca,'ytick', [1:5:20], 'yticklabel',{'20', '15','10','5'});
ylabel('neuron #');
set(gca,'clim', [0 50]);

%%  add colorbar
colorbar

%% plot counts

imagesc(tt*.1,1:20,log(yy(iiplot,:)')); axis xy;
%set(gca,'ytick', [1:5:20], 'yticklabel',{'20', '15','10','5'});
ylabel('neuron #');

%% add colorbar
colorbar


%% plot recovered latent by all three methods

lw = 2;
h = plot((tt)*.1, xx(iiplot), 'k',...
    (tt)*.1, xllemat(iiplot), '-', ...
    (tt)*.1, xpldsmat(iiplot), '-', ...
    (tt)*.1, xxsampmat(iiplot),'-', ...
    'linewidth', lw);
set(h(1), 'linewidth', 5);
set(h(2),'linewidth', 1);
set(gca,'ylim', ylm);
xlabel('time (s)');
ylabel('latent variable');


%%
xc = corrcoef(xx, xllemat).^2; r2s(1) = xc(2);
xc = corrcoef(xx, xpldsmat).^2; r2s(2) = xc(2);
xc = corrcoef(xx, xxsampmat).^2; r2s(3) = xc(2);

fprintf('---- R^2 vals -----\n');
fprintf('LLE: %.2f, PLDS: %.2f,  P-GPLVM: %.2f\n', r2s);
bar(r2s);
cc = get(gca, 'colororder');
hold on;
bar(2,r2s(2), 'facecolor', cc(2,:));
bar(3,r2s(3), 'facecolor', cc(3,:));
hold off;
box off;

%%  Plot recovered TCs

xgrid = gen_grid([min(xxsampmat(:,1)) max(xxsampmat(:,1))],50,nf); % x grid for plotting tuning curves
%xgrid = gen_grid([-3 3],50,nf); % x grid for plotting tuning curves
fftc = exp(get_tc(xxsampmat,result_la.ffmat,xgrid,result_la.rhoff,result_la.lenff));

neuronlist = 1:nneur;
ii = randperm(nneur);
lw = 2;
neuronlist = sort(neuronlist(ii(1:4)));
for ii=1:4
    neuronid = neuronlist(ii);
    subplot(2,2,ii), cla
    hold on,
    plot(simdata.xgrid,10*simdata.tuningCurve(:,neuronid),'k-', 'linewidth', lw);
    plot(xgrid,10*fftc(:,neuronid),'--', 'linewidth', 3, 'color', cc(3,:));
    
%     if ii==1
%         legend('true','estimate')
%     end
    title(['neuron ' num2str(neuronid)])
    set(gca,'xlim', [-3 3]);
    hold off
end

%% Plot all TCs

% Compute Poisson Linear Dynamic System (PLDS)
[xplds,params_plds] = run_plds(yy,nf);
xplds = xplds';
[xpldsmat,wplds] = align_xtrue(xplds,xx); % align the estimate with the true latent variable.

%%

xgrid = gen_grid([min(xxsampmat(:,1)) max(xxsampmat(:,1))],50,nf); % x grid for plotting tuning curves
%xgrid = gen_grid([-3 3],50,nf); % x grid for plotting tuning curves
fftc = exp(get_tc(xxsampmat,result_la.ffmat,xgrid,result_la.rhoff,result_la.lenff));

xgrid2 = gen_grid([min(xxsampmat(:,1))-.5 max(xxsampmat(:,1))+.5],50,nf); % x grid for plotting tuning curves
fftc_plds = exp(xgrid2*params_plds.model.C'/wplds(2)+repmat(params_plds.model.d'-params_plds.model.C'/wplds(2)*wplds(1),50,1));


%%
neuronlist = 1:nneur;
ii = randperm(nneur);
neuronlist = neuronlist(ii(1:4));

for ii=1:4
    neuronid = neuronlist(ii);
    subplot(2,2,ii), cla
    hold on,
    plot(simdata.xgrid,10*simdata.tuningCurve(:,neuronid),'k-', 'linewidth', 3);
    plot(xgrid2,10*fftc_plds(:,neuronid),'-', 'linewidth', 2.5, 'color', cc(2,:));
    plot(xgrid,10*fftc(:,neuronid),'-', 'linewidth', 2.5, 'color', cc(3,:));
%     plot(simdata.xgrid,simdata.tuningCurve(:,neuronid),'b-')
%     plot(xgrid,fftc(:,neuronid),'r-')
%     legend('true tc','estimated tc')
    title(['neuron ' num2str(neuronid)])
    hold off
    set(gca,'xlim', [-3 3],'ylim', [0 50]);
end




