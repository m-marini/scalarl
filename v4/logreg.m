function [Y1 m q] = logreg(X, Y)
  [Y1 m q] = linreg(X, exp(Y));
  Y1 = log(X * m + q);
endfunction
