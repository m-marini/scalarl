function Q = qFromTrace(X)
## -*- texinfo -*-
## @deftypefn  {Function File} @var{Q} = qFromTrace (@var{X})
## Return the policy values from trace data
## 
## @var{X} is the trace data
####
## The return values @var{Q} is a vector containing the policy values
## @end deftypefn
  Q = X(:, 10 : 17);
endfunction
