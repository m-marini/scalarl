function analyzeDiscreteAgent(
  X,                  # the indicators dump
  K0 = 0.7,           # the K threshold for C1 class
  Cd = 0.1,           # the C threshold for C0 class
  Ci = 0.5,           # the C threshold for C1 class
  EPS = 1e-3,         # the minimu J value to be considered
  PRC = 50 : 10 : 90, # the percentiles in the chart
  BINS = 20,          # the number of bins in histogram
  EPSH = 1)           # the optimal range of h to be considered
  
  NR = 2;             # number of rows
 
  # Number of steps, number of values
  [n m] = size(X);
  # Number of actors
  NA = floor((m - 2 ) / 2);
  NC = 1 + NA;

  JV = X(:, 1 : 2).^2;
  for actor = 0 : NA - 1
    j = actor * 2 + 3;
    JV = JV + X(:, j : j + 1).^2;
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
    j = actor * 2 + 3;
    col = actor + 2;
    
    GAMMAH = EPSH ./ prctile(X(:, j), PRC);
    GAMMAHM = ones(size(PRC)) * EPSH / mean(X(:, j));
    
    subplot(NR, NC, col);
    hist(X(:, j), BINS);
    title(sprintf("J h%d", actor));
    xlabel(sprintf("J h%d", actor));
    ylabel("# samples");
    grid on;

    subplot(NR, NC, col + NC);
    semilogy(PRC, [GAMMAH; GAMMAHM]);
    grid on;
    grid minor on;
    title(sprintf("gamma h%d", actor));
    xlabel("% corrected samples");
    ylabel(sprintf("gamma h%d", actor));

  endfor
endfunction
