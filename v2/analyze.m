function analyze(X)
  LAST = 30;
  NEPOCHS = max(X(:, 1)) + 1;
  NSTEPS = max(X(:, 2)) + 1;
  XAVG = mean(X(:, [3 4]));
  XLAST = mean(X(find(X(:, 1) > (NEPOCHS - LAST - 1)), [3 4]));
  
  Y = zeros(NSTEPS, 2);
  for i = 1 : NSTEPS
    Y(i, :) = mean(X(find(X(:, 2) == (i - 1)), [3 4]));
  endfor
    
  printf("    Rewards, Last 1000,     Error,  Last 100\n");
  printf("%9g, %9g, %9g, %9g\n",
     XAVG(1), XLAST(1), XAVG(2), XLAST(2));

  subplot(2,2,1);
  hist(X(:,3), 100);
  title("Rewards");
  grid on;

  subplot(2,2,2);
  hist(X(:,4), 100);
  title("Errors");
  grid on;
 
  subplot(2,2,3);
  plot([Y(:, 1), movingAvg(Y(:, 1), LAST)]);
  grid on;
  title("Returns");

  subplot(2,2,4);
  semilogy([Y(:, 2), movingAvg(Y(:, 2), LAST)]);
  grid on;
  title("Errorss");

endfunction
