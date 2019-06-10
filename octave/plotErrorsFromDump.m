function H = plotErrorsFromDump(X)
## -*- texinfo -*-
## @deftypefn  {Function File} @var{H} = plotErrorsFromDump(@var{X})
## Plot the errors from dump file
## 
## @var{X} is the dump data
##
## @var{H} is the graph handler
##
## @end deftypefn
  H = semilogy(X(:, 3));
  grid on;
  title("Errors");
endfunction
