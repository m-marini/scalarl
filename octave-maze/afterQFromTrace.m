function [ Q MASK ] = afterQFromTrace(X)
## -*- texinfo -*-
## @deftypefn  {Function File} [ @var{Q} @var{MASK} ] = afterQFromTrace(@var{X})
## Return the Q values and MASK for after state from trace data
## 
## @var{X} is the trace data
##
## The return values @var{Q} is the matrix containing Q value for after state
##
## The return values @var{MASK} is the matrix containing available action flags
## for after state
## @end deftypefn
  Q = X(:, 33 : 40);
  MASK = X(:, 49 : 55);
endfunction
