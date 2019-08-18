function V = expVFromTrace(X, LAMBDA=0.999)
## -*- texinfo -*-
## @deftypefn  {Function File} @var{V} = vFromTrace(@var{X}m @var{LAMBDA}=0.999)
## Return the expected V values from trace data
## 
## @var{X} is the trace data
##
## @var{LAMBDA} is the lambda discount of returns
####
## The return values @var{V} is the matrix containing expected V values
## @end deftypefn
  AF = afterVFromTrace(X);
  R = X(:, 4);
  V = AF * LAMBDA + R;
 endfunction
