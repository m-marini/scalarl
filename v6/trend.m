function Y = trend(X, LAST=100)
  NEPOCHS = max(X(:, 1)) + 1;
  NSTEPS = max(X(:, 2)) + 1;
  XAVG = mean(X(:, [3 4]));
  XLAST = mean(X(find(X(:, 1) > (NEPOCHS - LAST - 1)), [3 4]));
  
  Y = zeros(NSTEPS, 2);
  for i = 1 : NSTEPS
    Y(i, :) = mean(X(find(X(:, 2) == (i - 1)), [3 4]), 1);
  endfor
endfunction
