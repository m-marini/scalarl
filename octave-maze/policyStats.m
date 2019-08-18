function [N, M, AF, GF] = policyStats(X, H=10, W=10)
## -*- texinfo -*-
## @deftypefn  {Function File} [ @var{N} @var{M} @var{AF} @var{GF} ] = policyStats (@var{X}, @var(H), @var(W))
## Return the policy statistics from trace data
## 
## @var{X} is the trace data
## @var{H} is the maze height
## @var{W} is the maze width
##
## @var{N} is a matrix with the number of step in the maze
## @var{M} is a matrix with the max of available action in the maze
## @var{AF} is a matrix with the max frequency of action selection
## @var{GF} is a matrix with the max frequency of greedy action
## @end deftypefn
 N = zeros(H, W);
 M = zeros(H, W);
 AF = zeros(H, W);
 GF = zeros(H, W);
 POS = posFromTrace(X);
 A = actionIndexFromAction(actionsFromTrace(X));
 GA = actionIndexFromAction(greedyActionsFromTrace(X));
 Q = qFromTrace(X);
 for i = 1 : H
  for j = 1 : W
    IDX=find((POS(:, 1) == i-1) & (POS(:, 2) == j-1));
    n = size(IDX, 1);
    N(i, j) = n;
    if n > 0
      AF(i, j) = max(histc(A(IDX), 0 : 7));
      M(i, j) = sum(X(IDX(1), 34 : 41));
      GF(i, j) = max(histc(GA(IDX), 0 : 7));
    endif
  endfor
 endfor
endfunction
