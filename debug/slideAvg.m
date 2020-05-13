function [Y X] = slideAvg(Z, LEN = 10, STRIDE = 1)
  [N M] = size(Z);
  X = [LEN : STRIDE : N]';
  L = size(X, 1);
  Y = zeros(L, M);
  for i = 1 : L
    j = X(i);
    Y(i , :) = mean(Z(j - LEN + 1 : j, :), 1);
  endfor
endfunction
