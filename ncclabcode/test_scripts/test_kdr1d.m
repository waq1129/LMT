
%% Test Kernel Regression
%f = @(x) x.^2
f = @(x) 10*sin(.1*x)
N = 3e3;

dist_idx = 1;


%% One-Dimensional Example
mu = 30;
sigma = 30;
if(dist_idx)
    xx = normrnd(mu, sigma, N,1);   
else
    xx = exprnd(mu, N,1);
end
yy = f(xx) + randn(N,1);

fprintf('Exact Algorithm:\n')
tic 
[ey ex px] = kdr1d(xx,yy,300,[],'exact');
toc
fprintf('Approximate Algorithm:\n')
tic
[ay ax apx] = kdr1d(xx,yy,300,[]);
toc

figure(1);clf
subplot(211); hold on
[nn,nx] = hist(xx,ex);
bar(nx,nn/N)
plot(ex, px,'r','linewidth',2)
plot(ax,apx,'m','linewidth',2)
axis tight
axis square

subplot(212); hold on
plot(xx,yy,'.k')
plot(ex,f(ex),'b','linewidth',2)
plot(ex,ey,'r','linewidth',2)
plot(ax,ay,'m','linewidth',2)
axis tight
axis square

%% One-Dimensional Example
mu = 30;
sigma = 30;
if(dist_idx)
    xx = normrnd(mu, sigma, N,1);   
else
    xx = exprnd(mu, N,1);
end
yy = f(xx) + randn(N,1);


h0 = 1.06*(1/length(xx))^(1/5);
h = std(xx)*h0;   
hrng = h/3:h/5:1.5*h;

fprintf('Exact Algorithm:\n')
tic 
[ey ex px] = kdr1d(xx,yy,300,hrng,'exact');
toc
fprintf('Approximate Algorithm:\n')
tic
[ay ax apx] = kdr1d(xx,yy,300,hrng);
toc


figure(2);clf
subplot(211); hold on
[nn,nx] = hist(xx,ex);
bar(nx,nn/N)
plot(ex, px,'r','linewidth',2)
plot(ax,apx,'m','linewidth',2)
axis tight
axis square

subplot(212); hold on
plot(xx,yy,'.k')
plot(ex,f(ex),'b','linewidth',2)
plot(ex,ey,'r','linewidth',2)
plot(ax,ay,'m','linewidth',2)
axis tight
axis square


% 
% %% Two-Dimensional Example
% fprintf('\n\n2d example\n')
% mu = .1*rand(1,2);
% sigma = rand(2); sigma = sigma'*sigma;
% 
% if(1)
%     xx = mvnrnd(mu, sigma, N);   
% else
%     xx = exprnd(mu, N,1);
% end
% a = [.5; .5];
% yy = f(xx*a) + randn(N,1);
% 
% fprintf('Exact Algorithm:\n')
% tic 
% [ey ex px] = condexp2d(xx,yy,60,[],'exact');
% toc
% fprintf('Approximate Algorithm:\n')
% tic
% [ay ax apx] = condexp2d(xx,yy,60,[]);
% toc
% 
% figure(1);clf
% % subplot(211); hold on
% % [nn,nx] = hist(xx,ex);
% % bar(nx,nn/N)
% % plot(ex, px,'r')
% % plot(ax,apx,'m','linewidth',2)
% % axis tight
% % axis square
% 
% % subplot(212); hold on
% colormap gray
% subplot(311)
% imagesc(ex(:,1),ex(:,2),ey)
% colorbar
% axis tight
% axis square
% title('exact')
% subplot(312)
% imagesc(ax(:,1), ax(:,2), ay)
% colorbar
% axis tight
% axis square
% title('approx')
% subplot(313)
% imagesc(ax(:,1), ax(:,2), ey-ay)
% colorbar
% axis tight
% axis square
% title('diff')
% 
% fprintf('Mean Square Resid approx-exact: %0.2f\n', sum((ay(:)-ey(:)).^2))