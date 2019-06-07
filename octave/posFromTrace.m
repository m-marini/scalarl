function POS = posFromTrace(X)
## -*- texinfo -*-
## @deftypefn  {Function File} @var{POS} = posFromTrace (@var{X})
## Return the positions from trace data
## 
## @var{X} is the trace data
####
## The return values @var{POS} is a matrix containing the positions
## @end deftypefn
  POS = X(:, 6 : 7);
endfunction
