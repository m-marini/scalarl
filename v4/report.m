function report(X, LAST = 101, STRIDE = 100)
  N = size(X, 1);
  XR = [1 : STRIDE : N]';
  XX = [1 : N]';
  YR = movmean(X, LAST)(XR, :);

  REW = logreg(XX, X(:, 1))(XR, :);
  ERR = expreg(XX, X(:, 2))(XR, :);

  clf;  
  subplot(2,2,1);
  hist(X(:,1), 100);
  title("Rewards");
  grid on;

  subplot(2,2,2);
  hist(X(:,2), 100);
  title("Errors");
  grid on;
 
  subplot(2,2,3);
  plot(XR, [YR(:, 1) REW]);
  grid on;
  title("Rewards");

  subplot(2,2,4);
  semilogy(XR, [YR(:, 2) ERR]);
  grid on;
  title("Errors");

endfunction
