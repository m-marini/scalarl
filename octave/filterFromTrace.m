function Y = filterFromTrace(X, POS)
## -*- texinfo -*-
## @deftypefn  {Function File} [ @var{Y} ] = fitQFromTrace(@var{X}, @var{POS})
## Return the X values for the given position
## 
## @var{X} is the trace data
##
## @var{POS} is the position
##
## The return values @var{Y} is the trace data filtered for the given position
## @end deftypefn
  IDX = find(X(:, 6) == POS(1) & X(:, 7) == POS(2));
  Y = X(IDX, :);
endfunction
