function [R V] = statusFromTrace(X)
## -*- texinfo -*-
## @deftypefn  {Function File} [@var{R} @var[V]] = statusFromTrace (@var{X})
## Return the status from trace data
## 
## @var{X} is the trace data
####
## The return values @var{R} is a matrix containing the position rows
## The return values @var{V} is a matrix containing the speed rows
## @end deftypefn
  R = X(:, 20 : 22);
  V = X(:, 23 : 25);
endfunction
