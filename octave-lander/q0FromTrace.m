function Q = q0FromTrace(X)
## -*- texinfo -*-
## @deftypefn  {Function File} [@var{R} @var[V]] = qFromTrace (@var{X})
## Return the policy values from trace data
## 
## @var{X} is the trace data
####
## The return values @var{Q} is a matrix containing the policy rows
## @end deftypefn
  Q = X(:, 37 : 51);
endfunction
