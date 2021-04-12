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
    [DATA AA MODE] = regression(X(:, 1));
    REWARDS(:, (i - 1) * 2 + 2) = DATA(XR, :);
    LEGENDS{1, (i - 1) * 2 + 1} = [FILES(i, :) ""];
    LEGENDS{1, (i - 1) * 2 + 2} = [MODE " " FILES(i, :)];
#    ERRORS(:, (i - 1) * 2 + 2)  = expreg(XX, X(:, 2))(XR, :);
    [DATA AA MODE] = regression(X(:, 2));
    ERRORS(:, (i - 1) * 2 + 2)  = DATA(XR, :);
    LEGENDS{2, (i - 1) * 2 + 1} = [FILES(i, :) ""];
    LEGENDS{2, (i - 1) * 2 + 2} = [MODE " " FILES(i, :)];
  endfor
  
  clf;
  subplot(1,2,1);
  autoplot(XR, REWARDS);
  grid on;
  title("Rewards");
  legend(LEGENDS{1,:});

  subplot(1,2,2);
  semilogy(XR, ERRORS);
  grid on;
  title("Rewards");
  legend(LEGENDS{2,:});

endfunction
