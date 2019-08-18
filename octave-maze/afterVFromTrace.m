function V = afterVFromTrace(X)
## -*- texinfo -*-
## @deftypefn  {Function File} @var{V} = afterVFromTrace (@var{X})
## Return the V value for after state from trace data
## 
## @var{X} is the trace data
####
## The return values @var{V} is a matrix containing V value for after state
## @end deftypefn
  [Q MASK] = afterQFromTrace(X);
  V = vFromQMask(Q, MASK);
  ENDUP = X(:, 5) != 0;
  V(find(ENDUP)) = 0;
endfunction
