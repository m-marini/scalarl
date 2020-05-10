function actors(X, EPS=1e-3, PRC = 50 : 10 : 100, T=20)
  #EPS = 1e-3;
  #PRC = 50 : 10 : 100;
  #T = 20;
  
  # Number of steps
  n = size(X, 1);
  # Number of charts
  NC = 5;
  for actor = 1 : 3
    j = (actor - 1) * 2 + 3;
    i = (actor - 1) * NC + 1;
    
    XX = X(find(abs(X(:, j)) > EPS), j : j + 1);
    m = size(XX, 1);
    K = XX(:, 2) ./ XX(:, 1);
    KP = prctile(K, PRC);
    C = 100 ./ KP;
    e = sum(K >= 1);
    v = m - e;
    f = n - m;
    C1 = 100 ./ prctile(X(:, j), PRC);
    
    subplot(3, NC, i);
    hist(X(:, j), T);
    title(sprintf("Actor %d", actor));
    xlabel("J");
    ylabel("# samples");
    grid on;

    subplot(3, NC, i + 1);
    plot(PRC, C1);
    grid on;
    grid minor on;
    title(sprintf("Actor %d", actor));
    xlabel("% corrected samples");
    ylabel("Correction factor C1");

    subplot(3, NC, i + 2);
    hist(K, T);
    title(sprintf("Actor %d", actor));
    xlabel("K");
    ylabel("# samples");
    grid on;
    
    subplot(3, NC, i + 3);
    plot(PRC, C);
    grid on;
    grid minor on;
    title(sprintf("Actor %d", actor));
    xlabel("% corrected samples");
    ylabel("Correction factor C");

    subplot(3, NC, i + 4);
    pie([e v f]);
    title(sprintf("Actor %d", actor));
    #legend(["unoptimizable"; "optimizing"; "optimized"]);
 
  endfor
#endfunction
