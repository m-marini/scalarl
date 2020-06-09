function comparativeCharts(FILES = [
  "avg-11.csv",
  "avg-12.csv",
  "avg-13.csv",
  ],
  LENGTH = 100,
  STRIDE = 30)
  m = size(FILES, 1);

  X = csvread(FILES(1, :));
  N = size(X, 1);
  XR = [1 : STRIDE : N]';
  XX = [1 : N]';
  NR = size(XR, 1);
  
  REWARDS = zeros(NR, m * 2);
  ERRORS = zeros(NR, m * 2);
  LEGENDS = {};
  
  for i = 1 : m
    X = csvread(FILES(i, :));
    YR = movmean(X, LENGTH)(XR, :);
    REWARDS(:, (i - 1) * 2 + 1) = YR(:, 1);
    ERRORS(:, (i - 1) * 2 + 1) = YR(:, 2);
    REWARDS(:, (i - 1) * 2 + 2) = logreg(XX, X(:, 1))(XR, :);
    ERRORS(:, (i - 1) * 2 + 2)  = expreg(XX, X(:, 2))(XR, :);
    LEGENDS{(i - 1) * 2 + 1} = [FILES(i, :) ""];
    LEGENDS{(i - 1) * 2 + 2} = [FILES(i, :) " trend"];
  endfor
  
  LEGENDS
  
  clf;
  subplot(1,2,1);
  plot(XR, REWARDS);
  grid on;
  title("Rewards");
  legend(LEGENDS{1,:});

  subplot(1,2,2);
  semilogy(XR, ERRORS);
  grid on;
  title("Rewards");
  legend(LEGENDS{1,:});

endfunction
