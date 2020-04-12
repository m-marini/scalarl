function R = avgRewardFromTrace(X)
## -*- texinfo -*-
## @deftypefn  {Function File} [@var{R} @var[V]] = statusFromTrace (@var{X})
## Return the rewards from trace data
## 
## @var{X} is the trace data
####
## The return values @var{R} is a matrix containing the rewards rows
## @end deftypefn
  R = X(:, 32 : 34);
endfunction
