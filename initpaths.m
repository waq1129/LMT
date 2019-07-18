% initpaths.m
%
% Initialize paths 

addpath(genpath(pwd)); warning off

if ~exist('simdatadir','dir')
    mkdir simdatadir;
end