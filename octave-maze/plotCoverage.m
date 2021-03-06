function [N, M, AF, GF] = plotCoverage(X, H=10, W=10)
## -*- texinfo -*-
## @deftypefn  {Function File} [ @var{N} @var{M} @var{AF} @var{GF} ] = plotCoverage (@var{FNAME}, @var(H), @var(W))
## Plot the number of step in the maze
## 
## @var{X} is the trace data
##
## @var{H} is the maze height
##
## @var{W} is the maze width
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
 surface(N);
endfunction
