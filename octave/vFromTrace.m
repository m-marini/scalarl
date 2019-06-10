function V = vFromTrace(X)
## -*- texinfo -*-
## @deftypefn  {Function File} @var{V} = vFromTrace(@var{X})
## Return the V values from trace data
## 
## @var{X} is the trace data
####
## The return values @var{V} is the matrix containing V values
## @end deftypefn
  [Q MASK] = qFromTrace(X);
  V = vFromQMask(Q, MASK);
endfunction
