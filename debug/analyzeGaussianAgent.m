function analyzeGaussianAgent(X, EPS=1e-6, PRC = 50 : 10 : 90, T=20, EPSMU=1, EPSHS=1)
  #EPS = 1e-3;
  #PRC = 50 : 10 : 100;
  #T = 20;
  # Number of actors
  NA = 3;
  # Number of steps
  n = size(X, 1);
  # Number of charts per row
  NC = 7;
  NR = 3;

  JV = X(find(abs(X(:, 1)) >= EPS), 1 : 2).^2;
  # Number of valid steps
  cm = size(JV, 1);
  KV = JV(:, 2) ./ JV(:, 1);
  KVP = prctile(KV, PRC);
  ETAV = 1 ./ KVP;
  ce = sum(KV >= 1);
  cv = cm - ce;
  cf = n - cm;

  subplot(NR, NC, 1);
  pie([ce, cv, cf]);
  title("v");
  #legend("unoptimizable", "optimizing", "optimized");

  subplot(NR, NC, 1 + NC);
  plot(PRC, ETAV);
  grid on;
  grid minor on;
  title(sprintf("eta v"));
  xlabel("% corrected samples");
  ylabel("eta v");
  

  for actor = 1 : NA
    j = (actor - 1) * 4 + 3;
    i = (actor - 1) * 2 + 2;
    
    JMU = X(find(abs(X(:, j)) > EPS), j : j + 1).^2;
    JHS = X(find(abs(X(:, j)) > EPS), j + 2 : j + 3).^2;
    mmu = size(JMU, 1);
    mhs = size(JHS, 1);
    KMU = JMU(:, 2) ./ JMU(:, 1);
    KHS = JHS(:, 2) ./ JHS(:, 1);
    KMUP = prctile(KMU, PRC);
    KHSP = prctile(KHS, PRC);
    ETAMU= 1 ./ KMUP;
    ETAHS= 1 ./ KHSP;
    emu = sum(KMU >= 1);
    vmu = mmu - emu;
    fmu = n - mmu;
    GAMMAMU = EPSMU ./ prctile(X(:, j), PRC);
    
    subplot(NR, NC, i);
    pie([emu vmu fmu]);
    title(sprintf("mu %d", actor));

    subplot(NR, NC, i + NC);
    plot(PRC, ETAMU);
    #semilogy(PRC, C);
    grid on;
    grid minor on;
    title(sprintf("eta mu %d", actor));
    xlabel("% corrected samples");
    ylabel("eta mu");
 
    subplot(NR, NC, i + 2 * NC);
#    plot(PRC, GAMMAH);
    semilogy(PRC, GAMMAMU);
    grid on;
    grid minor on;
    title(sprintf("gamma mu %d", actor));
    xlabel("% corrected samples");
    ylabel("gamma mu");

    subplot(NR, NC, i + 1);
    pie([emu vmu fmu]);
    title(sprintf("hs %d", actor));

    subplot(NR, NC, i + NC + 1);
    plot(PRC, ETAMU);
    #semilogy(PRC, C);
    grid on;
    grid minor on;
    title(sprintf("eta hs %d", actor));
    xlabel("% corrected samples");
    ylabel("eta hs");
 
    subplot(NR, NC, i + 2 * NC + 1);
#    plot(PRC, GAMMAH);
    semilogy(PRC, GAMMAMU);
    grid on;
    grid minor on;
    title(sprintf("gamma hs %d", actor));
    xlabel("% corrected samples");
    ylabel("gamma hs");
  endfor
endfunction
