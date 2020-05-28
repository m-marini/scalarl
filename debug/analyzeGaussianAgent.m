function analyzeGaussianAgent(
  X,                  # the indicators dump
  K0 = 0.7,           # the K threshold for C1 class
  Cd = 0.1,           # the C threshold for C0 class
  Ci = 0.5,           # the C threshold for C1 class
  EPS = 1e-3,         # the minimu J value to be considered
  PRC = 50 : 10 : 90, # the percentiles in the chart
  BINS = 20,          # the number of bins in histogram
  EPSMU = 0.5,          # the averege range of delta mu
  EPSHS = 0.2)          # the average range of delta h sigma

  NR = 4;             # number of rows
  # Number of actors
  NA = floor((size(X, 2) - 2) / 4);
  # Number of steps
  n = size(X, 1);
  # Number of charts per row
  NC = NA + 1;
  JV = X(:, 1 : 2).^2;
  for actor = 0 : NA - 1
    j = actor * 2 + 3;
    JV = JV + X(:, j : j + 1).^2 + X(:, j + 2 : j + 3).^2;
  endfor
  JV = JV(find(abs(JV(:, 1)) >= EPS), 1 : 2);

  # Number of valid steps
  KV = JV(:, 2) ./ JV(:, 1);
  C0 = sum(KV > 1);
  C1 = sum(KV <= 1 & KV > K0);
  C2 = n - C0 - C1;

  if C0 > Cd * n
    advpie = [1 0 0];
    advice = "Reduce alpha";
    colorV = [0 1 0];
  elseif C1 >  Ci * n
    advpie = [0 1 0];
    advice = "Increment alpha";
    colorV = [1 1 0];
  else
    advpie = [0 0 1];
    advice = "No changes";
    colorV = [0 0 1];
  endif

  clf;
  subplot(NR, NC, 1);
  hist(KV, BINS);
  grid on;
  title(sprintf("Kv distribution"));
  xlabel("Kv");
  ylabel("# samples");
  
  subplot(NR, NC, 1 + NC);
  ph = pie([C0, C1, C2]);
  colormap([1 0 0; 1 1 0; 0 1 0]);
  title("Step classes");
  text(-1, -1.25, sprintf("Advice: %s", advice));

  for actor = 0 : NA - 1
    j = actor * 4 + 3;
    col = actor + 2;
    
    XMU = X(find(X(:, j) > EPS), j);
    XHS = X(find(X(:, j + 2) > EPS), j + 2);
    JMU = XMU .^2;
    JHS = XHS .^2;
    PCMU = prctile(XMU, PRC);
    GAMMAMU = EPSMU ./ PCMU;
    GAMMAMUM = ones(size(GAMMAMU)) * EPSMU ./ mean(XMU);
    GAMMAHS = EPSHS ./ prctile(XHS, PRC);
    GAMMAHSM = ones(size(GAMMAHS)) * EPSHS ./ mean(XHS);
    
    subplot(NR, NC, col);
    hist(JMU, BINS);
    grid on;
    title(sprintf("J mu%d distribution", actor));
    xlabel(sprintf("J mu%d distribution", actor));
    ylabel("# samples");

    subplot(NR, NC, col + NC);
#    plot(PRC, GAMMAH);
    semilogy(PRC, [GAMMAMU; GAMMAMUM]);
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
#    plot(PRC, GAMMAH);
    semilogy(PRC, [GAMMAHS; GAMMAHSM]);
    grid on;
    grid minor on;
    title(sprintf("gamma hs %d", actor));
    xlabel("% corrected samples");
    ylabel(sprintf("gamma hs %d", actor));

  endfor
endfunction
