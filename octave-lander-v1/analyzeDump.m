function analyzeDump(X)
  L = 100;
  NL = 100;
  avg = mean(X, 1);
  lastEpisode = max(X(:, 2));
  LAST = X( find(X(:, 2) >= lastEpisode - NL), :);
  last100 = mean(LAST, 1);
  
  lastEpoch = max(X(1, 2));
  
  Y = zeros(lastEpisode, 5);
  for i = 1 : lastEpisode
    Y(i, :) = mean( X( find( X(:, 2) == i), :), 1);
  endfor
  
  printf("    Steps,  Last 100,   Returns, Last 1000,     Error,  Last 100\n");
  printf("%9g, %9g, %9g, %9g, %9g, %9g\n",
    avg(3), last100(3), avg(4), last100(4), avg(5), last100(5));

  subplot(2,3,1);
  hist(X(:,3));
  title("Steps");
  grid on;
  
  subplot(2,3,2);
  hist(X(:,4));
  title("Returns");
  grid on;
  
  subplot(2,3,3);
  hist(X(:,5));
  title("Error");
  grid on;
  
  subplot(2,3,4);
  semilogy([Y(:, 3), movingAvg(Y(:, 3), L)]);
  grid on;
  title("Steps");
  
  subplot(2,3,5);
  plot([Y(:, 4), movingAvg(Y(:, 4), L)]);
  grid on;
  title("Return");
  
  subplot(2,3,6);
  semilogy([Y(:, 5), movingAvg(Y(:, 5), L)]);
  grid on;
  title("Errors");

endfunction
