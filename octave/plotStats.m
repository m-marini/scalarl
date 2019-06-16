function plotStats(FNAMES, TITLE, LEGEND, COL = 1)
## -*- texinfo -*-
## @deftypefn  {Function File} [ @var{RETURNS} @var{ERRORS} ] = plotStats(@var{FNAMES}, @var(TITLE), @var(LEGEND=, @var(COL) = 1)
## Plot the statistica by hyer parameters
## 
## @var{FNAMEs} is the stats folders
##
## @var{TITLE} is the title of graph
##
## @var{LEGEND} is the legend for each hyper parameter value
##
## @var{COL} 1 for MEAN, 6 for MEDIAN
## @end deftypefn
  [RETURNS ERRORS] = compareStats(FNAMES, COL);
  
  subplot(1, 2, 1);
  plot(RETURNS);
  grid on;
  title([TITLE " Returns" ]);
  legend(LEGEND);
  
  subplot(1, 2, 2);
  semilogy(ERRORS);
  grid on;
  title([TITLE " Errors" ]);
  legend(LEGEND);
  
endfunction
