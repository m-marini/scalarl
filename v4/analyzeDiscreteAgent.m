function analyzeDiscreteAgent(
  X,                  # the indicators dump
  EPSH = 0.24,           # the optimal range of h to be considered
  K0 = 0.7,           # the K threshold for C1 class
  EPS = 100e-6,       # the minimu J value to be considered
  PRC = [50 : 10 : 90]', # the percentiles in the chart
  BINS = 20)          # the number of bins in histogram
  # Number of direction
  NDIR = 8;
  # Number of h Jet
  NHJET = 3;
  # Number of z Jet
  NZJET = 5;
  # Number of steps, number of values
  [n m] = size(X);
  # Number of actors
  NA = 3;
  #DR = 100:102;

  # Compute J value of network (sum over the J values of critics and actors)
  V0 = X(:, 1);
  VSTAR = X(:, 2);
  V1 = X(:, 3);
  RPI = X(:, 4);
  TD = (VSTAR - V0) .^ 2;
  J = X(:, 5);
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
  # Number of otpimizing steps
  C1 = sum(K <= 1 & K > K0);
  # Number of optimizied steps
  C2 = n - C0 - C1;
  
  # Direction actor
  hChart = {};
  alpha = X(1, 7);
  # Computes the J values
  H = X(:, 8 : 15);
  HSTAR = X(:, 16 : 23);
  DH = HSTAR - H;
  DH2 = DH .^ 2;
  DH2M = mean(DH2, 2);
  hChart{1 , 1} = DH2M;
  # Compute alpha for percetile
  PCH = prctile(sqrt(DH2M), PRC);
  hChart{1, 2} = EPSH ./ PCH * alpha;
  # Compute mean
  hChart{1, 3} = ones(size(PRC, 1), 1) * EPSH ./ sqrt(mean(DH2M)) * alpha;

  # H power actor
  alpha = X(1, 32);
  # Computes the J values
  H = X(:, 33 : 35);
  HSTAR = X(:, 36 : 38);
  DH = HSTAR - H;
  DH2 = DH .^ 2;
  DH2M = mean(DH2, 2);
  hChart{2 , 1} = DH2M;
  # Compute alpha for percetile
  PCH = prctile(sqrt(DH2M), PRC);
  hChart{2, 2} = EPSH ./ PCH * alpha;
  # Compute mean
  hChart{2, 3} = ones(size(PRC, 1), 1) * EPSH ./ sqrt(mean(DH2M)) * alpha;

  # Z power actor
  alpha = X(1, 42);
  # Computes the J values
  H = X(:, 43 : 47);
  HSTAR = X(:, 48 : 52);
  DH = HSTAR - H;
  DH2 = DH .^ 2;
  DH2M = mean(DH2, 2);
  hChart{3 , 1} = DH2M;
  # Compute alpha for percetile
  PCH = prctile(sqrt(DH2M), PRC);
  hChart{3, 2} = EPSH ./ PCH * alpha;
  # Compute mean
  hChart{3, 3} = ones(size(PRC, 1), 1) * EPSH ./ sqrt(mean(DH2M)) * alpha;
 
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
  autoplot([RPI RTREND]);
  grid on;
  title(sprintf("Average Reward\n%s Trend", RMODE));
  ylabel("Reward");
  xlabel("Step");
  
  subplot(NR, NC, 2 + NC);
  autoplot([TD TDTREND]);
  grid on;
  title(sprintf("Squared TD Error\n%s Trend", TDMODE));
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
    autoplot(PRC, [hChart{actor + 1, 2}, hChart{actor + 1, 3}]);
    grid on;
    grid minor on;
    title(sprintf("alpha %d", actor));
    xlabel("% corrected samples");
    ylabel(sprintf("alpha %d", actor));
  endfor

endfunction
