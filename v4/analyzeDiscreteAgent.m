function analyzeDiscreteAgent(
  X,                      # the indicators dump
  EPSH = 0.24,            # the optimal range of h to be considered
  K0 = 0.7,               # the K threshold for C1 class
  EPS = 100e-6,           # the minimu J value to be considered
  PRC = [50 : 10 : 90]',  # the percentiles in the chart
  BINS = 20,              # the number of bins in histogram
  NPOINTS = 300,          # the number of point for historical charts
  LENGTH = 300            # the number of averaging samples for historical chrts
  )
  
  # Indices constants
  IV0 = 1;
  IVStar = 2;
  IV1 = 3;
  IRPi = 4;
  IJ = 5;
  IJ1 = 6;
  IAlphaDir = 7;
  IHDir = [8 : 15];
  IHDirStar = [16 : 23];
  IAlphaH = 32;
  IHH = [33 : 35];
  IHHStar = [36 : 38];
  IAlphaZ = 42;
  IHZ = [43 : 47];
  IHZStar = [48 : 52];

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
  V0 = X(:, IV0);
  VSTAR = X(:, IVStar);
  V1 = X(:, IV1);
  RPI = X(:, IRPi);
  TD = (VSTAR - V0) .^ 2;
  J = X(:, IJ);
  J1 = X(:, IJ1);
  
  # Filter the J values greater than threshold EPS
  VI = find(abs(J) >= EPS);
  J = J(VI, 1);
  J1 = J1(VI, 1);
  # Compute the K values
  K = J1 ./ J;
  
  # Number of invalid steps
  C0 = sum(K > 1);
  # Number of optimizing steps
  C1 = sum(K <= 1 & K > K0);
  # Number of optimizied steps
  C2 = n - C0 - C1;
  
  # Class percentages
  RED = C0 / (C0 + C1 + C2);
  YELLOW = C1 / (C0 + C1 + C2);
  GREEN = C2 / (C0 + C1 + C2);
  
  # Direction actor
  hChart = {};
  alpha = X(1, IAlphaDir);
  # Computes the J values
  H = X(:, IHDir);
  HSTAR = X(:, IHDirStar);
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
  alpha = X(1, IAlphaH);
  # Computes the J values
  H = X(:, IHH);
  HSTAR = X(:, IHHStar);
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
  alpha = X(1, IAlphaZ);
  # Computes the J values
  H = X(:, IHZ);
  HSTAR = X(:, IHZStar);
  DH = HSTAR - H;
  DH2 = DH .^ 2;
  DH2M = mean(DH2, 2);
  hChart{3 , 1} = DH2M;
  # Compute alpha for percetile
  PCH = prctile(sqrt(DH2M), PRC);
  hChart{3, 2} = EPSH ./ PCH * alpha;
  # Compute mean
  hChart{3, 3} = ones(size(PRC, 1), 1) * EPSH ./ sqrt(mean(DH2M)) * alpha;

  csvwrite("td.csv", TD);
  
  [TDX TD TDTREND TDMODE] = meanchart(TD, NPOINTS, LENGTH);
  [RPIX RPI RTREND RMODE] = meanchart(RPI, NPOINTS, LENGTH);
 
  NR = 2;             # number of rows
  NC = 2 + NA;

  clf;

  subplot(NR, NC, 1);
  autoplot(RPIX, [RPI RTREND]);
  grid on;
  title(sprintf("Average Reward\n%s Trend", RMODE));
  ylabel("Reward");
  xlabel("Step");
  
  subplot(NR, NC, 1 + NC);
  pie([C0, C1, C2]);
  colormap([1 0 0; 1 1 0; 0 1 0]);
  title("Step classes");
  
  subplot(NR, NC, 2);
  autoplot(TDX, [TD TDTREND]);
  grid on;
  title(sprintf("Squared TD Error\n%s Trend", TDMODE));
  ylabel("delta^2");
  xlabel("Step");

  subplot(NR, NC, 2 + NC);
  hist(K, BINS);
  grid on;
  title(sprintf("K distribution"));
  xlabel("K");
  ylabel("# samples");
  
  for actor = 0 : NA - 1
    col = actor + 3;
    subplot(NR, NC, col);
    autoplot(PRC, [hChart{actor + 1, 2}, hChart{actor + 1, 3}]);
    grid on;
    grid minor on;
    title(sprintf("alpha %d", actor));
    xlabel("% corrected samples");
    ylabel(sprintf("alpha %d", actor));

    subplot(NR, NC, col + NC);
    hist(hChart{actor + 1, 1}, BINS);
    grid on;
    title(sprintf("J %d distribution", actor));
    xlabel(sprintf("J %d distribution", actor));
    ylabel("# samples");
  endfor
  
  printf("ANN\n");
  printf("%s rewards trend from %.1f to %.1f\n", RMODE, RTREND(1), RTREND(end));
  printf("%s MSE (RMSE) trend from %.1f (%.1f) to %.1f (%.1f)\n",
    TDMODE, TDTREND(1), sqrt(TDTREND(1)), TDTREND(end), sqrt(TDTREND(end)));
  printf("%.0f%% red class\n", RED * 100);
  printf("%.0f%% yellow class\n", YELLOW * 100);
  printf("%.0f%% green class\n", GREEN * 100);
    
  printf("Optimal actor alpha: ");
  for actor = 0 : NA - 1
    if actor > 0
      printf(", ");
    endif
    printf("%.1e", hChart{actor + 1, 3}(1));
  endfor
  printf("\n");
  #Optimal actor alphas: 96 mU, 72 mU, 85 mU.

endfunction
