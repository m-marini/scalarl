function actors(X, EPS=1e-3, PRC = 50 : 10 : 90, T=20)
  #EPS = 1e-3;
  #PRC = 50 : 10 : 100;
  #T = 20;
  # Number of actors
  NA = 3;
  # Number of steps
  n = size(X, 1);
  # Number of charts
  NC = 5;

  CX = X(find(abs(X(:, 1)) >= EPS), 1 : 2);
  # Number of vaid steps
  cm = size(CX, 1);
  CK = CX(:, 2) ./ CX(:, 1);
  CKP = prctile(CK, PRC);
  CC = 1 ./ CKP;
  ce = sum(CK >= 1);
  cv = cm - ce;
  cf = n - cm;

  subplot(NA + 1, NC, 1);
  #hist(CK, T);
  grid on;
  title(sprintf("Critic K"));
  xlabel("K");
  ylabel("# samples");
  
  subplot(NA + 1, NC, 2);
  plot(PRC, CC);
  grid on;
  grid minor on;
  title(sprintf("Critic C"));
  xlabel("% corrected samples");
  ylabel("Correction factor C");
  
  subplot(NA + 1, NC, 3);
  pie([ce, cv, cf]);
  title("Critic Steps");
  #legend("unoptimizable", "optimizing", "optimized");

  for actor = 1 : NA
    j = (actor - 1) * 2 + 3;
    i = actor * NC + 1;
    
    XX = X(find(abs(X(:, j)) > EPS), j : j + 1);
    m = size(XX, 1);
    K = XX(:, 2) ./ XX(:, 1);
    KP = prctile(K, PRC);
    C = 1 ./ KP;
    e = sum(K >= 1);
    v = m - e;
    f = n - m;
    C1 = 1 ./ prctile(X(:, j), PRC);
    
    subplot(NA + 1, NC, i + 0);
    hist(K, T);
    title(sprintf("Actor %d K", actor));
    xlabel("K");
    ylabel("# samples");
    grid on;
    
    subplot(NA + 1, NC, i + 1);
    plot(PRC, C);
    #semilogy(PRC, C);
    grid on;
    grid minor on;
    title(sprintf("Actor %d C", actor));
    xlabel("% corrected samples");
    ylabel("Correction factor C");

    subplot(NA + 1, NC, i + 2);
    pie([e v f]);
    title(sprintf("Actor %d Steps", actor));
    #legend(["unoptimizable"; "optimizing"; "optimized"]);
 
    
    subplot(NA + 1, NC, i + 3);
    hist(X(:, j), T);
    title(sprintf("Actor %d J", actor));
    xlabel("J");
    ylabel("# samples");
    grid on;

    subplot(NA + 1, NC, i + 4);
    plot(PRC, C1);
    semilogy(PRC, C1);
    grid on;
    grid minor on;
    title(sprintf("Actor %d C1", actor));
    xlabel("% corrected samples");
    ylabel("Correction factor C1");

  endfor
endfunction
