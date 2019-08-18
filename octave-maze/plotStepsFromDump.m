function H = plotStepsFromDump(X)
## -*- texinfo -*-
## @deftypefn  {Function File} @var{H} = plotStepsFromDump(@var{X})
## Plot the step count from dump file
## 
## @var{X} is the dump data
##
## @var{H} is the graph handler
##
## @end deftypefn
  H = semilogy(X(:, 1));
  grid on;
  title("Steps");
endfunction
