function analyze(X, LAST = 100)
  Y = trend(X, LAST);
  [YR, XR] = slideAvg(Y, LAST);

  subplot(2,2,1);
  hist(X(:,3), 100);
  title("Rewards");
  grid on;

  subplot(2,2,2);
  hist(X(:,4), 100);
  title("Errors");
  grid on;
 
  subplot(2,2,3);
  plot(XR, YR(:, 1));
  grid on;
  title("Rewards");

  subplot(2,2,4);
  semilogy(XR, YR(:, 2));
  grid on;
  title("Errors");

endfunction
