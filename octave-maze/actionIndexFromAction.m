function IDX = actionIndexFromAction(A)
## -*- texinfo -*-
## @deftypefn  {Function File} @var{IDX} = actionIndexFromAction (@var{A})
## Return the action index from action vector
## 
## @var{A} is the action vector
##
## @var{IDX} is a vector with action indices
## @end deftypefn
 N = size(A, 1);
 IDX = zeros(N, 1);
 for i = 1 : N
   IDX(i) = find(A(i, :));
 endfor
endfunction
