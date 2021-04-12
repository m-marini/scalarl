function analyzeGaussianAgent(
  X,                  # the indicators dump
  EPSMU = [10e-3 62.5e-3 250e-3],        # the averege range of delta mu
  EPSHS = [347e-3 347e-3 347e-3],        # the average range of delta h sigma
  K0 = 0.7,           # the K threshold for C1 class
  EPS = 10e-6,         # the minimu J value to be considered
  PRC = [50 : 10 : 90]', # the percentiles in the chart
  BINS = 20)          # the number of bins in histogram
  
  # Number of colums per actors
  NCOLA = 8;
  # Number of steps, number of values
  [n m] = size(X);
  # Number of actors
  NA = floor((m - 6) / NCOLA);
  DR = 1:3;

  # Extract the state value estimation for the state
  V0 = X(:, 1);
  # Extract the target state value estimation for the state
  VSTAR = X(:, 2);
  # Extract the state value estimation for the next state
  V1 = X(:, 3);
  # Extract the average state value
  RPI = X(:, 4);
  TD = (VSTAR - V0) .^ 2;
  # Compute J value of network (mean square error over outputs before training)
  J = X(:, 5);
  # Compute J' value of network  (mean square error over outputs after training)
  J1 = X(:, 6);
 
  # Filter the J values greater than threshold EPS
  VI = find(abs(J) >= EPS);
  J = J(VI, 1);
  J1 = J1(VI, 1);
  # Compute the K values
  K = J1 ./ J;
  [TDTREND, TDP, TDMODE] = regression(TD);
  [RTREND, RP, RMODE] = regression(RPI);

  # Number of invalid steps
  C0 = sum(K > 1);
  # Number of optimizing steps
  C1 = sum(K <= 1 & K > K0);
  # Number of optimizied steps
  C2 = n - C0 - C1;

  muChart = {};
  sigmaChart = {};
  for actor = 0 : NA - 1
    j = actor * NCOLA;
    alphaMu = X(1, j + 7);
    
    # Computes mu
    MU = X(:, j + 8);
    MUSTAR = X(:, j + 9);
    
    DMU = MUSTAR - MU;
    DMU2 = DMU .^ 2;

    muChart{actor + 1 , 1} = DMU2;

    # Compute alpha for percetile
    PCMU = prctile(sqrt(DMU2), PRC);
    muChart{actor + 1, 2} = EPSMU(actor + 1) ./ PCMU * alphaMu;

    # Compute mean
    muChart{actor + 1, 3} = ones(size(PRC, 1), 1) * EPSMU(actor + 1) ./ sqrt(mean(DMU2)) * alphaMu;

    alphaHs = X(1, j + 11);
    # Computes the J values
    HS = X(:, j + 12);
    HSSTAR = X(:, j + 13);
    
    DHS = HSSTAR - HS;
    DHS2 = DHS .^ 2;

    sigmaChart{actor + 1 , 1} = DHS2;

    # Compute alpha for percetile
    PCHS = prctile(sqrt(DHS2), PRC);
    sigmaChart{actor + 1, 2} = EPSHS(actor + 1) ./ PCHS * alphaHs;

    # Compute mean
    sigmaChart{actor + 1, 3} = ones(size(PRC, 1), 1) * EPSHS(actor + 1) ./ sqrt(mean(DHS2)) * alphaHs;
  endfor
  
  NR = 4;
  NC = 1 + NA;

  clf;
  subplot(NR, NC, 1);
  pie([C0, C1, C2]);
  colormap([1 0 0; 1 1 0; 0 1 0]);
  title("Step classes");
  
  subplot(NR, NC, 1 + NC);
  hist(K, BINS);
  grid on;
  title(sprintf("K distribution"));
  xlabel("K");
  ylabel("# samples");
  
  subplot(NR, NC, 1 + 2 * NC);
  autoplot([RTREND]);
  grid on;
  title(sprintf("Average Reward\n%s Trend", RMODE));
  ylabel("Reward");
  xlabel("Step");
  
  subplot(NR, NC, 1 + 3 * NC);
  autoplot([TDTREND]);
  grid on;
  title(sprintf("Squared TD Error\n%s Trend", TDMODE));
  ylabel("\\delta^2");
  xlabel("Step");

  for actor = 0 : NA - 1
    col = actor + 2;
    
    subplot(NR, NC, col);
    hist(muChart{actor + 1, 1}, BINS);
    grid on;
    title(sprintf("J \\mu%d distribution", actor));
    xlabel(sprintf("J \\mu%d", actor));
    ylabel("# samples");

    subplot(NR, NC, col + NC);
    autoplot(PRC, [muChart{actor + 1, 2}, muChart{actor + 1, 3}]);
    grid on;
    grid minor on;
    title(sprintf("\\alpha_\\mu %d", actor));
    xlabel("% corrected samples");
    ylabel(sprintf("\\alpha_\\mu %d", actor));

    subplot(NR, NC, col + 2 * NC);
    hist(sigmaChart{actor + 1, 1}, BINS);
    grid on;
    title(sprintf("J h%d distribution", actor));
    xlabel(sprintf("J h %d", actor));
    ylabel("# samples");

    subplot(NR, NC, col + 3 * NC);
    autoplot(PRC, [sigmaChart{actor + 1, 2}, sigmaChart{actor + 1, 3}]);
    grid on;
    grid minor on;
    title(sprintf("\\alpha_h %d", actor));
    xlabel("% corrected samples");
    ylabel(sprintf("\\alpha_h %d", actor));

  endfor
endfunction
