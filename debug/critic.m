function critic(X, EPS = 1e-3, T = 20)
  EPS = 1e-3;
  T = 20;
  
  # Number of steps
  n = size(X, 1);
  # Valid step data
  XX = X(find(abs(X(:, 3)) >= EPS), 3 : 4);
  # Number of vaid steps
  m = size(XX, 1);
  K = XX(:, 2) ./ XX(:, 1);
  KP = prctile(K, 10 : 10 : 90);
  C = 100 ./ KP;
  e = sum(K >= 1);
  v = m - e;
  f = n - m;
  
  
  subplot(1, 3, 1);
  hist(K, T);
  grid on;
  title(sprintf("Critic K"));
  xlabel("K");
  ylabel("# samples");
  
  subplot(1, 3, 2);
  plot([10:10:90],KP);
  grid on;
  grid minor on;
  xlabel("% corrected samples");
  ylabel("Correction factor C");
  
  subplot(1, 3, 3);
  pie([e, v, f]);
  title("Steps");
  legend("unoptimizable", "optimizing", "optimized");
  
endfunction
