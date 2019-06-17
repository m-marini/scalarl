function plotStats(FNAMES, TITLE, LEGEND, COL = 1, PGN_PREFIX = "")
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
  
  plot(RETURNS);
  grid on;
  grid minor on;
  title([TITLE " Returns" ]);
  legend(LEGEND, "location", "southeast");
  xlabel("Episodes");
  ylabel("Returns");
  print([PGN_PREFIX, TITLE, "-returns.png"]);
  
  semilogy(ERRORS);
  grid on;
  grid minor on;
  title([TITLE " Errors" ]);
  legend(LEGEND, "location", "southeast");
  xlabel("Episodes");
  ylabel("Errors");
  print([PGN_PREFIX TITLE "-errors.png"]);
  
endfunction
