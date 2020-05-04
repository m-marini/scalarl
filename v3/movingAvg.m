function Y = movingAvg(X, L)
  [N M] = size(X);
  Y = zeros([N M]);
  for i = 1 : min(L, N)
    Y(i , :) = mean(X(1 : i, :), 1);
  endfor
  for i = L + 1 : N
    Y(i , :) = mean(X(i - L + 1 : i, :), 1);
  endfor
endfunction
