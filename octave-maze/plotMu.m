function [MU N, M, AF, GF] = plotMu(X, W=10, H=10)
## -*- texinfo -*-
## @deftypefn  {Function File} [ @var{MU} @var{N} @var{M} @var{AF} @var{GF} ] = plotMu (@var{FNAME}, @var(H), @var(W))
## Plot the mu value in the maze
## 
## @var{FNAME} is the trace filename
##
## @var{H} is the maze height
##
## @var{W} is the maze width
##
## @var{MU} is a matrix with the mu value
##
## @var{N} is a matrix with the number of step in the maze,
##
## @var{M} is a matrix with the max of available action in the maze,
##
## @var{AF} is a matrix with the max frequency of action selection,
##
## @var{GF} is a matrix with the max frequency of greedy action,
## @end deftypefn
 [N, M, AF, GF] = policyStats(X, H, W);
 F = floor((N + M - 1) ./ M);
 MU = (AF ./ F - 1);
 MU = MU .* (MU >= 0);
 surface(MU);
 endfunction
