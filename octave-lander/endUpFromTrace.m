function EU = endUpFromTrace(X)
## -*- texinfo -*-
## @deftypefn  {Function File} @var{EU} = endUpFromTrace (@var{X})
## Return the status from trace data
## 
## @var{X} is the trace data
####
## The return values @var{EU} is a row vector containing the end up flag
## @end deftypefn
  EU = X(:, 19);
endfunction
