function A = actionsFromTrace(X)
## -*- texinfo -*-
## @deftypefn  {Function File} @var{GA} = actionsFromTrace (@var{X})
## Return the actions from trace data
## 
## @var{X} is the trace data
####
## The return values @var{GA} is a vector containing the action
## @end deftypefn
  A = X(:, 3);
endfunction
