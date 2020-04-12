function H = plotSteps(X, L=30)
## -*- texinfo -*-
## @deftypefn  {Function File} @var{H} = plotStepsFromDump(@var{X})
## Plot the step count from dump file
## 
## @var{X} is the dump data
##
## @var{H} is the graph handler
##
## @end deftypefn
  H = semilogy(movingAvg(X(:, 1), L));
  grid on;
  title("Steps");
endfunction
