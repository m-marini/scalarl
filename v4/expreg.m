function [Y1 m q] = expreg(X, Y)
  [Y1 m q] = linreg(X, log(Y));
  Y1 = exp(X * m + q);
endfunction
