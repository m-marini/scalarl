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
  GA = zeros(N, 8);
  [Q0 MASK] = qFromTrace(X);
  for I = 1 : N
   IDX = find(MASK(I, :));
   Q = Q0(I, IDX);
   [_ A] = max(Q, [], 2);
   GA(I, IDX(A)) = 1;
  endfor
endfunction
