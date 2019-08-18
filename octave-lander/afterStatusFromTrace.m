function [R V] = afterStatusFromTrace(X)
## -*- texinfo -*-
## @deftypefn  {Function File} [@var{R} @var[V]] = afterStatusFromTrace (@var{X})
## Return the result status from trace data
## 
## @var{X} is the trace data
####
## The return values @var{R} is a matrix containing the position rows
## The return values @var{V} is a matrix containing the speed rows
## @end deftypefn
  R = X(:, 26 : 28);
  V = X(:, 29 : 31);
endfunction
