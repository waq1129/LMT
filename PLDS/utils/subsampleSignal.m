function uSub = subsampleSignal(u,dt);
%
% function uSub = subsampleSignal(u,dt);
%

uSub = [u];
T    = size(uSub,2); 
Tf   = floor(T/dt)*dt;
uSub = reshape(uSub(:,1:Tf),size(uSub,1),dt,Tf/dt);
uSub = squeeze(uSub(:,1,:));
