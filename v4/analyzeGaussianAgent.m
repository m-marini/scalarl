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
  NA = floor((m - 2 ) / 4);
  NC = 1 + NA;

  # Compute J value of network (sum over the J values of critics and actors)
  J = X(:, 1 : 2).^2;
  for actor = 0 : NA - 1
    j = actor * 2 + 3;
    J = J + X(:, j : j + 1).^2 + X(:, j + 2 : j + 3).^2;
  endfor
  # Filter the J values greater than threshold EPS
  J = J(find(abs(J(:, 1)) >= EPS), 1 : 2);
  # Compute the K values
  K = J(:, 2) ./ J(:, 1);
  KTREND = expRegression(K);
  J1TREND = expRegression(J(:, 2));

  size(K, 1)
  
  # Number of invalid steps
  C0 = sum(K > 1);
  # Number of otpimizing steps
  C1 = sum(K <= 1 & K > K0);
  # Number of optimizied steps
  C2 = n - C0 - C1;

  muChart = {};
  hsChart = {};
  for actor = 0 : NA - 1
    j = actor * 4 + 3;
    # Computes the J mu values
    muChart{actor + 1 , 1} =  X(:, j) .^2;
    # Compute Gamma mu for percetile
    PCMU = prctile(X(:, j), PRC);
    muChart{actor + 1, 2} = EPSMU ./ PCMU;
    # Coumpute mean
    muChart{actor + 1, 3} = ones(size(PRC, 1), 1) * EPSMU ./ mean(X(:, j));
       
    # Computes the J mu values
    hsChart{actor + 1 , 1} =  X(:, j + 2) .^2;
    # Compute Gamma mu for percetile
    PCHS = prctile(X(:, j + 2), PRC);
    hsChart{actor + 1, 2} = EPSHS ./ PCHS;
    # Coumpute mean
    hsChart{actor + 1, 3} = ones(size(PRC, 1), 1) * EPSHS ./ mean(X(:, j + 2));
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
  plot([KTREND]);
  grid on;
  title(sprintf("K trend"));
  ylabel("K");
  xlabel("Step");
  
  subplot(NR, NC, 1 + 3 * NC);
  semilogy([J1TREND]);
  grid on;
  title(sprintf("Prediction error trend", actor));
  ylabel("J'");
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
    title(sprintf("gamma mu %d", actor));
    xlabel("% corrected samples");
    ylabel(sprintf("gamma mu %d", actor));

    subplot(NR, NC, col + NC * 2);
    hist(JHS, BINS);
    grid on;
    title(sprintf("J hs%d distribution", actor));
    xlabel(sprintf("J hs%d distribution", actor));
    ylabel("# samples");

    subplot(NR, NC, col + NC * 3);
    semilogy(PRC, [hsChart{actor + 1, 2} hsChart{actor + 1, 3}]);
    grid on;
    grid minor on;
    title(sprintf("gamma hs %d", actor));
    xlabel("% corrected samples");
    ylabel(sprintf("gamma hs %d", actor));

  endfor
endfunction
