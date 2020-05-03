function actors(X, EPS=1e-3, T=20)
  # Number of steps
  n = size(X, 1);
  for actor = 1 : 3
    j = (actor - 1) * 2 + 3;
    i = (actor - 1) * 3 + 1;
    
    XX = X(find(abs(X(:, j)) > EPS), j : j + 1);
    m = size(XX, 1);
    K = XX(:, 2) ./ XX(:, 1);
    C = 100 / max(K);
    e = sum(K >= 1);
    v = m - e;
    f = n - m;
    C1 = 100 / mean(X(:, j));
    
    subplot(3, 3, i);
    hist(X(:, j), T);
    title(sprintf("Actor %d J, C = %.1f%%", actor, C1));
    grid on;

    subplot(3, 3, i + 1);
    hist(K, T);
    title(sprintf("Actor %d K, C = %.1f%%", actor, C));
    grid on;
    
    subplot(3, 3, i + 2);
    pie([e v f]);
    title("Steps");
    #legend(["unoptimizable"; "optimizing"; "optimized"]);
 
  endfor
endfunction
