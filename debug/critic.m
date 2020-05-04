function critic(X, EPS = 1e-3, T=20)
  # Number of steps
  n = size(X, 1);
  # Valid step data
  X = X(find(abs(X(:, 1)) >= EPS), :);
  # Number of vaid steps
  m = size(X, 1);
  K = X(:, 2) ./ X(:, 1);
  C = 100 / max(K);
  e = sum(K >= 1);
  v = m - e;
  f = n - m;
  
  subplot(1, 2, 1);
  hist(K, T);
  grid on;
  title(sprintf("Critic K, C = %.1f%%", C));
  
  subplot(1, 2, 2);
  pie([e, v, f]);
  title("Steps");
  legend("unoptimizable", "optimizing", "optimized");
  
endfunction
