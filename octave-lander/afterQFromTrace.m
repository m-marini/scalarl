function Q = afterQFromTrace(X)
## -*- texinfo -*-
## @deftypefn  {Function File} [@var{R} @var[V]] = qFromTrace (@var{X})
## Return the policy values for the next state from trace data
## 
## @var{X} is the trace data
####
## The return values @var{Q} is a matrix containing the policy rows
## @end deftypefn
  Q = X(:, 47 : 61);
endfunction
