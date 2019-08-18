function [RETURNS ERRORS] = compareStats(FNAMES, COL = 1)
## -*- texinfo -*-
## @deftypefn  {Function File} [ @var{RETURNS} @var{ERRORS} ] = compareStats(@var{FNAMES}, @var(COL))
## Load the stats files for hyper parameters comparison
## 
## @var{FNAMEs} is the stats folders
##
## @var{COL} 1 for MEAN, 6 for MEDIAN
##
## @var{RETURNS} is the matrix with the returns data by hyper parameters
##
## @var{ERRORS} is the matrix with the errors data by hyper parameters
## @end deftypefn
  N = size(FNAMES, 1);
  M = 0;
  for i = 1 : N
    FNAME = deblank(FNAMES(i, :));
    X = csvread([FNAME "/returns.csv"]);
    Y = csvread([FNAME "/errors.csv"]);
    L = size(X, 1);
    if (i == 1)
      RETURNS(:, i) = X(:, COL);
      ERRORS(:, i) = Y(:, COL);
      M = L;
    elseif (L < M)
      RETURNS = [RETURNS(1 : L, :) X(:, COL)];
      ERRORS = [ERRORS(1 : L, :) Y(:, COL)];
      M = L;
    else
      RETURNS = [RETURNS(:, :) X(1 : M, COL)];
      ERRORS = [ERRORS(:, :) Y(1 : M, COL)];
    endif
  endfor
endfunction
