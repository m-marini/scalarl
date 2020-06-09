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
  NA = floor((m - 2 ) / 2);

  # Compute J value of network (sum over the J values of critics and actors)
  J = X(:, 1 : 2).^2;
  for actor = 0 : NA - 1
    j = actor * 2 + 3;
    J = J + X(:, j : j + 1).^2;
  endfor
  # Filter the J values greater than threshold EPS
  J = J(find(abs(J(:, 1)) >= EPS), 1 : 2);
  # Compute the K values
  K = J(:, 2) ./ J(:, 1);
  KTREND = expRegression(K);
  J1TREND = expRegression(J(:, 2));
  
  # Number of invalid steps
  C0 = sum(K > 1);
  # Number of otpimizing steps
  C1 = sum(K <= 1 & K > K0);
  # Number of optimizied steps
  C2 = n - C0 - C1;
  
  hChart = {};
  for actor = 0 : NA - 1
    j = actor * 2 + 3;
    # Computes the J mu values
    hChart{actor + 1 , 1} =  X(:, j) .^2;
    # Compute Gamma mu for percetile
    PCH = prctile(X(:, j), PRC);
    hChart{actor + 1, 2} = EPSH ./ PCH;
    # Compute mean
    hChart{actor + 1, 3} = ones(size(PRC, 1), 1) * EPSH ./ mean(X(:, j));
  endfor
 
  NR = 2;             # number of rows
  NC = 2 + NA;

  clf;
  subplot(NR, NC, 1);
  hist(K, BINS);
  grid on;
  title(sprintf("K distribution"));
  xlabel("K");
  ylabel("# samples");
  
  subplot(NR, NC, 1 + NC);
  pie([C0, C1, C2]);
  colormap([1 0 0; 1 1 0; 0 1 0]);
  title("Step classes");
  
  subplot(NR, NC, 2);
  plot([KTREND]);
  grid on;
  title(sprintf("K trend"));
  ylabel("K");
  xlabel("Step");
  
  subplot(NR, NC, 2 + NC);
  semilogy([J1TREND]);
  grid on;
  title(sprintf("Prediction error trend", actor));
  ylabel("J'");
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
    title(sprintf("gamma mu %d", actor));
    xlabel("% corrected samples");
    ylabel(sprintf("gamma mu %d", actor));
  endfor

endfunction
