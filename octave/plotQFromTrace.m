function Q = plotQFromTrace(X, POS, W = 10, H = 10)
## -*- texinfo -*-
## @deftypefn  {Function File} @var{Q} = plotVFromTrace(@var{X}, @var{POS}, @var{W} = 10, @var{H} = 10)
## Plot the Q value of a cell from trace data
## 
## @var{X} is the trace data
##
## @var{POS} is the cell position
##
## @var{W} is the maze width
##
## @var{H} is the maze height
##
## @var{Q} the Q values
## @end deftypefn
  N = W * H;
  NA = 8;
  IDX = find(X(:, 6) == POS(1) & X(:, 7) == POS(2));
  Q = X(IDX, 10 : 17);
  H = plot(Q);
  legend(
   ["Q(" int2str(POS) ", 0)"],
   ["Q(" int2str(POS) ", 1)"],
   ["Q(" int2str(POS) ", 2)"],
   ["Q(" int2str(POS) ", 3)"],
   ["Q(" int2str(POS) ", 4)"],
   ["Q(" int2str(POS) ", 5)"],
   ["Q(" int2str(POS) ", 6)"],
   ["Q(" int2str(POS) ", 7)"]
   );
  title(["Q(" int2str(POS) ")"]);
  grid on;
endfunction
