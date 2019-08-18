function [Q MASK] = qFromTrace(X)
## -*- texinfo -*-
## @deftypefn  {Function File} [ @var{Q} @var{MASK} ] = qFromTrace(@var{X})
## Return the Q values and MASK from trace data
## 
## @var{X} is the trace data
##
## The return values @var{Q} is the matrix containing the Q values
##
## The return values @var{MASK} is the matrix containing the available actions
## @end deftypefn
  Q = X(:, 17 : 24);
  MASK = X(:, 41 : 48);
endfunction
