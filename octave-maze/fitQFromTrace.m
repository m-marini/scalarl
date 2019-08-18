function [Q MASK] = fitQFromTrace(X)
## -*- texinfo -*-
## @deftypefn  {Function File} [ @var{Q} @var{MASK} ] = fitQFromTrace(@var{X})
## Return the Q values after fitting and MASK from trace data
## 
## @var{X} is the trace data
##
## The return values @var{Q} is the matrix containing the Q values after fitting
##
## The return values @var{MASK} is the matrix containing the available actions
## @end deftypefn
  Q = X(:, 25 : 32);
  MASK = X(:, 41 : 48);
endfunction
