function [Y1 m q] = linreg(X, Y)
  Sxy = cov(X, Y);
  Sxx = cov(X);
  m = Sxy / Sxx;
  q = mean(Y) - m * mean(X);
  Y1 = X * m + q;
endfunction
