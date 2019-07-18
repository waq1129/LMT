function [M,Mdim] = ndgraphLapl(nn)
% NDGRAPHLAPL - generates matrix for ND graph Laplacian
% 
% [M,Mdim] = ndgraphLapl(nn)

% Generates matrix M such that vec(x)'*M*vec(x) is the sum of squared
% pairwise differences between entries in a x.
%
% Inputs:
%  nn (ndimensions x 1) = [n1; n2; n3; ... ]  number of coefficients along
%                         each tensory dimension
%  
% Outputs:
%  M (prod(nn) x prod(nn) = graph Laplacian matrix
%  Mdim = matrices for computing pairwise diffs along each tensor
%         dimension separately (useful if you want to penalize different
%         entries differently)
% 
%  M = Mdim{1} + Mdim{2} + Mdim{3} + ...
%
%  EXAMPLE SCRIPT
%  ---------------
%  nn = [10 5 3]; x = randn(nn);  
%  %% Compute row, column & depth squared diffs%%
%  coldiffs = sum(sum(sum(diff(x).^2)));
%  rowdiffs = sum(sum(sum(diff(permute(x,[2 1 3])).^2))); 
%  depthdiffs = sum(sum(sum(diff(permute(x,[3 1 2])).^2))); 
%  %% Now compute using graph Laplacian %%
%  [M,Mdim] = ndgraphLapl(nn);
%  [x(:)'*Mdim{1}*x(:) coldiffs]
%  [x(:)'*Mdim{2}*x(:) rowdiffs]
%  [x(:)'*Mdim{3}*x(:) depthdiffs]
%
%  $Id$

ntdim = length(nn);  % tensor dimension
nelts = prod(nn);    % number of total coefficients

% Make sparse matrix for each tensor dimension
for j = 1:ntdim
    ectr = [1; 2*ones(nn(j)-2,1); 1];
    eoff = -ones(nn(j),1);
    L{j} = spdiags([eoff, ectr, eoff],-1:1, nn(j),nn(j));
end

% Combine these using Kronecker product
M = sparse(nelts,nelts);
for j = 1:ntdim
    nbefore = prod(nn(1:(j-1))); % number of elements in previous tensor dimension
    nafter = prod(nn(j+1:end)); % number of elements in later tensor dimensions
    Mdim{j} = kron(L{j},speye(nbefore));
    Mdim{j} = kron(speye(nafter),Mdim{j});
    
    M = M + Mdim{j};  % Add them up
    
end
