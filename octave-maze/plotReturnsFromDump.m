function H = plotReturnsFromDump(X)
## -*- texinfo -*-
## @deftypefn  {Function File} @var{H} = plotReturnsFromDump (@var{X})
## Plot the returns from dump file
## 
## @var{X} is the dump data
##
## @var{H} is the graph handler
##
## @end deftypefn
  H = plot(X(:, 2));
  grid on;
  title("Returns");
endfunction
