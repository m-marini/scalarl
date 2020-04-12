function H = plotReturns(X, L=30)
## -*- texinfo -*-
## @deftypefn  {Function File} @var{H} = plotStepsFromDump(@var{X})
## Plot the step count from dump file
## 
## @var{X} is the dump data
##
## @var{H} is the graph handler
##
## @end deftypefn
  H = plot(movingAvg(X(:, 2), L));
  grid on;
  title("Return");
endfunction
