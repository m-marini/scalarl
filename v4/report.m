function report(X, LAST = 100, STRIDE = 100)
  [YR, XR] = slideAvg(X, LAST, STRIDE);

  YM = ones(size(YR)) .* mean(X, 1);
    
  subplot(2,2,1);
  hist(X(:,1), 100);
  title("Rewards");
  grid on;

  subplot(2,2,2);
  hist(X(:,2), 100);
  title("Errors");
  grid on;
 
  subplot(2,2,3);
  plot(XR, [YR(:, 1) YM(:, 1)]);
  grid on;
  title("Rewards");

  subplot(2,2,4);
  semilogy(XR, [YR(:, 2), YM(:, 2)]);
  grid on;
  title("Errors");

endfunction
