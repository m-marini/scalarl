function analyzeGaussianAgent(
  X,                  # the indicators dump
  K0 = 0.7,           # the K threshold for C1 class
  EPS = 10e-6,         # the minimu J value to be considered
  PRC = [50 : 10 : 90]', # the percentiles in the chart
  BINS = 20,          # the number of bins in histogram
  EPSMU = 0.5,        # the averege range of delta mu
  EPSHS = 0.2)        # the average range of delta h sigma

  NR = 4;             # number of rows
  
  # Number of steps, number of values
  [n m] = size(X);
  # Number of actors
  NA = floor((m - 3 ) / 6);
  NC = 1 + NA;

  # Compute J value of network (sum over the J values of critics and actors)
  J = X(:, 1 : 2).^2;
  TD = J(:, 1);
  for actor = 0 : NA - 1
    j = actor * 6 + 4;
    J = J + X(:, j + 1 : j + 2).^2 + X(:, j + 4 : j + 5).^2;
  endfor
  # Filter the J values greater than threshold EPS
  J = J(find(abs(J(:, 1)) >= EPS), 1 : 2);
  # Compute the K values
  K = J(:, 2) ./ J(:, 1);
  TDTREND = expRegression(TD);
  RTREND = logRegression(X(:, 3));

  size(K, 1);
  
  # Number of invalid steps
  C0 = sum(K > 1);
  # Number of otpimizing steps
  C1 = sum(K <= 1 & K > K0);
  # Number of optimizied steps
  C2 = n - C0 - C1;

  muChart = {};
  hsChart = {};
  for actor = 0 : NA - 1
    j = actor * 6 + 4;
    alphaMu = X(1, j);
    # Computes the J mu values
    muChart{actor + 1 , 1} =  X(:, j) .^2;
    # Compute Gamma mu for percetile
    PCMU = prctile(X(:, j + 1), PRC);
    muChart{actor + 1, 2} = EPSMU ./ PCMU * alphaMu;
    # Coumpute mean
    muChart{actor + 1, 3} = ones(size(PRC, 1), 1) * EPSMU ./ mean(X(:, j)) * alphaMu;
    
    alphaHs = X(1, j + 3);
    # Computes the J mu values
    hsChart{actor + 1 , 1} =  X(:, j + 4) .^2;
    # Compute Gamma mu for percetile
    PCHS = prctile(X(:, j + 4), PRC);
    hsChart{actor + 1, 2} = EPSHS ./ PCHS * alphaHs;
    # Coumpute mean
    hsChart{actor + 1, 3} = ones(size(PRC, 1), 1) * EPSHS ./ mean(X(:, j + 4)) * alphaHs;
  endfor
  
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
  
  subplot(NR, NC, 1 + 2 * NC);
  plot([RTREND]);
  grid on;
  title(sprintf("Average Reward trend"));
  ylabel("Reward");
  xlabel("Step");
  
  subplot(NR, NC, 1 + 3 * NC);
  semilogy([TDTREND]);
  grid on;
  title(sprintf("Squared TD error trend"));
  ylabel("delta^2");
  xlabel("Step");
  
  for actor = 0 : NA - 1
    col = actor + 2;
    subplot(NR, NC, col);
    hist(muChart{actor + 1, 1}, BINS);
    grid on;
    title(sprintf("J mu%d distribution", actor));
    xlabel(sprintf("J mu%d distribution", actor));
    ylabel("# samples");

    subplot(NR, NC, col + NC);
    semilogy(PRC, [muChart{actor + 1, 2}, muChart{actor + 1, 3}]);
    grid on;
    grid minor on;
    title(sprintf("alpha mu %d", actor));
    xlabel("% corrected samples");
    ylabel(sprintf("alpha mu %d", actor));

    subplot(NR, NC, col + NC * 2);
    hist(hsChart{actor + 1, 1}, BINS);
    grid on;
    title(sprintf("J hs%d distribution", actor));
    xlabel(sprintf("J hs%d distribution", actor));
    ylabel("# samples");

    subplot(NR, NC, col + NC * 3);
    semilogy(PRC, [hsChart{actor + 1, 2} hsChart{actor + 1, 3}]);
    grid on;
    grid minor on;
    title(sprintf("alpha hs %d", actor));
    xlabel("% corrected samples");
    ylabel(sprintf("alpha hs %d", actor));

  endfor
endfunction
