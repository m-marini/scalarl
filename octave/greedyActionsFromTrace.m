function GA = greedyActionsFromTrace(X)
## -*- texinfo -*-
## @deftypefn  {Function File} @var{GA} = greedyActionsFromTrace (@var{X})
## Return the greedy actions from trace data
## 
## @var{X} is the trace data
####
## The return values @var{GA} is a vector containing the greedy action
## @end deftypefn
  N = size(X, 1);
  GA = zeros(N, 1);
  for i = 1 : N
   IDX = find(X(i, 34 : 41));
   Q = X(i, 10 : 17)(IDX);
   [_ A] = max(Q, [], 2);
   GA(i) = IDX(A) - 1;
  endfor
endfunction
