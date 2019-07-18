function M = spblkdiag(blocks)
%function M = spblkdiag(blocks)
%
% function generating efficiently a sparse matrix containing
% subblocks blocks(:,:,i) as block i along the diagonal.
% this is 1000 times faster than blkdiag!!!!
%
% Bernard Haasdonk 31.8.2009

% This program is open source.  For license terms, see the COPYING file.
%
% --------------------------------------------------------------------
% ATTRIBUTION NOTICE:
% This product includes software developed for the RBmatlab project at
% (C) Universities of Stuttgart and MÃ?nster, Germany.
%
% RBmatlab is a MATLAB software package for model reduction with an
% emphasis on Reduced Basis Methods. The project is maintained by
% M. Dihlmann, M. Drohmann, B. Haasdonk, M. Ohlberger and M. Schaefer.
% For Online Documentation and Download we refer to www.morepas.org.
% --------------------------------------------------------------------


[n,m,k] = size(blocks);

row_ind = (1:n)'*ones(1,m*k);
row_offs =(ones(n*m,1)*(0:(k-1)))*n;
row_ind = row_ind(:)+row_offs(:);

col_ind = ones(n,1)*(1:(m*k));
col_ind = col_ind(:);

M = sparse(row_ind,col_ind,blocks(:));%| \docupdate 