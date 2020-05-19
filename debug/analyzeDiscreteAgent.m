function analyzeDiscreteAgent(X, EPS=1e-3, PRC = 50 : 10 : 90, T=20, EPSH=1)
  #EPS = 1e-3;
  #PRC = 50 : 10 : 100;
  #T = 20;
  # Number of actors
  NA = 3;
  # Number of steps
  n = size(X, 1);
  # Number of charts
  NC = 5;

  JV = X(find(abs(X(:, 1)) >= EPS), 1 : 2).^2;
  # Number of valid steps
  cm = size(JV, 1);
  KV = JV(:, 2) ./ JV(:, 1);
  KVP = prctile(KV, PRC);
  ETAV = 1 ./ KVP;
  ce = sum(KV >= 1);
  cv = cm - ce;
  cf = n - cm;

  subplot(NA + 1, NC, 1);
  #hist(CK, T);
  grid on;
  title(sprintf("Kv"));
  xlabel("Kv");
  ylabel("# samples");
  
  subplot(NA + 1, NC, 2);
  plot(PRC, ETAV);
  grid on;
  grid minor on;
  title(sprintf("\eta v"));
  xlabel("% corrected samples");
  ylabel("Correction factor alpha v");
  
  subplot(NA + 1, NC, 3);
  pie([ce, cv, cf]);
  title("Critic Steps");
  #legend("unoptimizable", "optimizing", "optimized");

  for actor = 1 : NA
    j = (actor - 1) * 2 + 3;
    i = actor * NC + 1;
    
    JH = X(find(abs(X(:, j)) > EPS), j : j + 1).^2;
    m = size(JH, 1);
    KH = JH(:, 2) ./ JH(:, 1);
    KHP = prctile(KH, PRC);
    ETAH= 1 ./ KHP;
    e = sum(KH >= 1);
    v = m - e;
    f = n - m;
    GAMMAH = EPSH ./ prctile(X(:, j), PRC);
    
    subplot(NA + 1, NC, i + 0);
    hist(KV, T);
    title(sprintf("Actor %d Kh", actor));
    xlabel("Kh");
    ylabel("# samples");
    grid on;
    
    subplot(NA + 1, NC, i + 1);
    plot(PRC, ETAH);
    #semilogy(PRC, C);
    grid on;
    grid minor on;
    title(sprintf("Actor %d eta h", actor));
    xlabel("% corrected samples");
    ylabel("eta h");

    subplot(NA + 1, NC, i + 2);
    pie([e v f]);
    title(sprintf("Actor %d Steps", actor));
    #legend(["unoptimizable"; "optimizing"; "optimized"]);
 
    subplot(NA + 1, NC, i + 3);
    hist(X(:, j).^2, T);
    title(sprintf("Actor %d Jh", actor));
    xlabel("Jh");
    ylabel("# samples");
    grid on;

    subplot(NA + 1, NC, i + 4);
#    plot(PRC, GAMMAH);
    semilogy(PRC, GAMMAH);
    grid on;
    grid minor on;
    title(sprintf("Actor %d gamma h", actor));
    xlabel("% corrected samples");
    ylabel("gamma h");

  endfor
endfunction
