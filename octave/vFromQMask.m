function V = vFromQMask(Q, MASK)
## -*- texinfo -*-
## @deftypefn  {Function File} @var{V} = vFromQMask(@var{Q}, @var{MASK})
## Return the V values from Q MASK
## 
## @var{Q} is the matrix containing Q values
##
## @var{MASK} is the matrix containing the available action flags
##
## The return values @var{V} is the matrix containing V values
## @end deftypefn
  N = size(Q, 1);
  V = zeros(N, 1);
  for i = 1 : N
    IDX = find(MASK(i, :) != 0);
    V(i) = max(Q(i, IDX));
  endfor
endfunction
