function analyzeGaussianAgent(
  X,                  # the indicators dump
  EPSMU = 0.5,        # the averege range of delta mu
  EPSHS = 0.2,        # the average range of delta h sigma
  K0 = 0.7,           # the K threshold for C1 class
  EPS = 10e-6,         # the minimu J value to be considered
  PRC = [50 : 10 : 90]', # the percentiles in the chart
  BINS = 20)          # the number of bins in histogram
  
  # Number of colums per actors
  NCOLA = 8;
  # Number of steps, number of values
  [n m] = size(X);
  # Number of actors
  NA = floor((m - 4 ) / NCOLA);
  DR = 1:3;

  # Compute J value of network (sum over the J values of critics and actors)
  V0 = X(:, 1);
  VSTAR = X(:, 2);
  V1 = X(:, 3);
  RPI = X(:, 4);
  TD = (VSTAR - V0) .^ 2;
  J = TD;
  J1 = (VSTAR - V1) .^ 2;

  for actor = 0 : NA - 1
    j = actor * NCOLA + 5;
    HMU = X(:, j + 1);
    HMUSTAR = X(:, j + 2);
    HMU1 = X(:, j + 3);
    
    HS = X(:, j + 5);
    HSSTAR = X(:, j + 6);
    HS1 = X(:, j + 7);

    JHMU = (HMUSTAR - HMU) .^ 2;
    JHMU1 = (HMUSTAR - HMU1) .^ 2;

    JHS = (HSSTAR - HS) .^ 2;
    JHS1 = (HSSTAR - HS1) .^ 2;

    J = J + JHMU + JHS;
    J1 = J1 + JHMU1 + JHS1;
   
  endfor

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
  # Number of otpimizing steps
  C1 = sum(K <= 1 & K > K0);
  # Number of optimizied steps
  C2 = n - C0 - C1;
  
  muChart = {};
  sigmaChart = {};
  for actor = 0 : NA - 1
    j = actor * NCOLA + 5;
    alphaMu = X(1, j);
    # Computes the J values
    HMU = X(:, j + 1);
    HMUSTAR = X(:, j + 2);
    
    DHMU = HMUSTAR - HMU;
    DHMU2 = DHMU .^ 2;

    muChart{actor + 1 , 1} = DHMU2;

    # Compute alpha for percetile
    PCHMU = prctile(sqrt(DHMU2), PRC);
    muChart{actor + 1, 2} = EPSMU ./ PCHMU * alphaMu;

    # Compute mean
    muChart{actor + 1, 3} = ones(size(PRC, 1), 1) * EPSMU ./ sqrt(mean(DHMU2)) * alphaMu;

    alphaHs = X(1, j + 4);
    # Computes the J values
    HS = X(:, j + 5);
    HSSTAR = X(:, j + 6);
    
    DHS = HSSTAR - HS;
    DHS2 = DHS .^ 2;

    sigmaChart{actor + 1 , 1} = DHS2;

    # Compute alpha for percetile
    PCHS = prctile(sqrt(DHS2), PRC);
    sigmaChart{actor + 1, 2} = EPSHS ./ PCHS * alphaHs;

    # Compute mean
    sigmaChart{actor + 1, 3} = ones(size(PRC, 1), 1) * EPSHS ./ sqrt(mean(DHS2)) * alphaHs;
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
    hist(muChart{actor + 1, 1}, BINS);
    grid on;
    title(sprintf("J h%d distribution", actor));
    xlabel(sprintf("J h %d", actor));
    ylabel("# samples");

    subplot(NR, NC, col + 3 * NC);
    autoplot(PRC, [muChart{actor + 1, 2}, muChart{actor + 1, 3}]);
    grid on;
    grid minor on;
    title(sprintf("\\alpha_h %d", actor));
    xlabel("% corrected samples");
    ylabel(sprintf("\\alpha_h %d", actor));

  endfor
endfunction
