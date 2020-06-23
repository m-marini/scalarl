function analyzeDiscreteAgent(
  X,                  # the indicators dump
  K0 = 0.7,           # the K threshold for C1 class
  EPS = 1e-3,         # the minimu J value to be considered
  PRC = [50 : 10 : 90]', # the percentiles in the chart
  BINS = 20,          # the number of bins in histogram
  EPSH = 1)           # the optimal range of h to be considered
  
  # Number of steps, number of values
  [n m] = size(X);
  # Number of actors
  NA = floor((m - 3 ) / 3);

  # Compute J value of network (sum over the J values of critics and actors)
  TD = X(:, 1 : 2).^2;
  J = TD;
  for actor = 0 : NA - 1
    j = actor * 3 + 4;
    J = J + X(:, j + 1 : j + 2).^2;
  endfor
  # Filter the J values greater than threshold EPS
  J = J(find(abs(J(:, 1)) >= EPS), 1 : 2);
  # Compute the K values
  K = J(:, 2) ./ J(:, 1);
  TDTREND = expRegression(TD(:, 1));
  RTREND = logRegression(X(:, 3));
  
  # Number of invalid steps
  C0 = sum(K > 1);
  # Number of otpimizing steps
  C1 = sum(K <= 1 & K > K0);
  # Number of optimizied steps
  C2 = n - C0 - C1;
  
  hChart = {};
  for actor = 0 : NA - 1
    alpha = X(1, j);
    j = actor * 3 + 4;
    # Computes the J values
    hChart{actor + 1 , 1} =  X(:, j + 1) .^2;
    # Compute Gamma for percetile
    PCH = prctile(X(:, j + 1), PRC);
    hChart{actor + 1, 2} = EPSH ./ PCH * alpha;
    # Compute mean
    hChart{actor + 1, 3} = ones(size(PRC, 1), 1) * EPSH ./ mean(X(:, j + 1)) * alpha;
  endfor
 
  NR = 2;             # number of rows
  NC = 2 + NA;

  clf;
  subplot(NR, NC, 1);
  pie([C0, C1, C2]);
  colormap([1 0 0; 1 1 0; 0 1 0]);
  title("Step classes");
  
  subplot(NR, NC, 2);
  hist(K, BINS);
  grid on;
  title(sprintf("K distribution"));
  xlabel("K");
  ylabel("# samples");
  
  subplot(NR, NC, 1 + NC);
  plot([RTREND]);
  grid on;
  title(sprintf("Average reward"));
  ylabel("Reward");
  xlabel("Step");
  
  subplot(NR, NC, 2 + NC);
  semilogy([TDTREND]);
  grid on;
  title(sprintf("Squared TD Error trend", actor));
  ylabel("delta^2");
  xlabel("Step");
  
  for actor = 0 : NA - 1
    col = actor + 3;
    subplot(NR, NC, col);
    hist(hChart{actor + 1, 1}, BINS);
    grid on;
    title(sprintf("J %d distribution", actor));
    xlabel(sprintf("J %d distribution", actor));
    ylabel("# samples");

    subplot(NR, NC, col + NC);
    semilogy(PRC, [hChart{actor + 1, 2}, hChart{actor + 1, 3}]);
    grid on;
    grid minor on;
    title(sprintf("alpha %d", actor));
    xlabel("% corrected samples");
    ylabel(sprintf("alpha %d", actor));
  endfor

endfunction
