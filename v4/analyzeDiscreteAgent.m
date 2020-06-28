function analyzeDiscreteAgent(
  X,                  # the indicators dump
  EPSH = 1,           # the optimal range of h to be considered
  K0 = 0.7,           # the K threshold for C1 class
  EPS = 100e-6,       # the minimu J value to be considered
  PRC = [50 : 10 : 90]', # the percentiles in the chart
  BINS = 20)          # the number of bins in histogram
  # Number of actions
  NAC = 5;
  # Number of colums per actors
  NCOLA = 16;
  # Number of steps, number of values
  [n m] = size(X);
  # Number of actors
  NA = floor((m - 4 ) / NCOLA);
  #DR = 100:102;

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
    IH = j + 1;
    IHSTAR = j + NAC + 1;
    IH1 = j + 2 * NAC + 1;
    H = X(:, IH : IH + NAC - 1);
    HSTAR = X(:, IHSTAR : IHSTAR + NAC - 1);
    H1 = X(:, IH1 : IH1 + NAC - 1);
    JH = (HSTAR - H) .^ 2;
    JH1 = (HSTAR - H1) .^ 2;
    J = J + sum(JH, 2);
    J1 = J1 + sum(JH1, 2);
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
  
  hChart = {};
  for actor = 0 : NA - 1
    j = actor * 16 + 5;
    alpha = X(1, j);
    # Computes the J values
    IH = j + 1;
    IHSTAR = IH + NAC;
    H = X(:, IH : IH + NAC - 1);
    HSTAR = X(:, IHSTAR : IHSTAR + NAC - 1);
    
    DH = HSTAR - H;
    DH2 = DH .^ 2;
    DH2M = mean(DH2, 2);

    hChart{actor + 1 , 1} = DH2M;

    # Compute alpha for percetile
    PCH = prctile(sqrt(DH2M), PRC);
    hChart{actor + 1, 2} = EPSH ./ PCH * alpha;

    # Compute mean
    hChart{actor + 1, 3} = ones(size(PRC, 1), 1) * EPSH ./ sqrt(mean(DH2M)) * alpha;
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
  autoplot([RTREND]);
  grid on;
  title(sprintf("Average Reward\n%s Trend", RMODE));
  ylabel("Reward");
  xlabel("Step");
  
  subplot(NR, NC, 2 + NC);
  autoplot([TDTREND]);
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
