function [V, Q] = plotVFromTrace(X, POS, W = 10, H = 10)
## -*- texinfo -*-
## @deftypefn  {Function File} @var{H} = plotVFromTrace(@var{X}, @var{POS}, @var{W} = 10, @var{H} = 10)
## Plot the V value of a cell from trace data
## 
## @var{X} is the trace data
##
## @var{POS} is the cell position
##
## @var{W} is the maze width
##
## @var{H} is the maze height
##
## @var{V} the V values
## @end deftypefn
 N = W * H;
  NA = 8;
  IDX = find(X(:, 6) == POS(1) & X(:, 7) == POS(2));
  Q = X(IDX, 10 : 17);
  V = max(Q, [], 2);
  plot(V);
  legend(["V(" int2str(POS) ")"]);
  title(["V(" int2str(POS) ")"]);
  grid on;
endfunction
