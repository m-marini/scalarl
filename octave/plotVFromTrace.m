function H = plotVFromTrace(file, POS, W = 10, H = 10)
  N = W * H;
  NA = 8;
  X = csvread(file);
  IDX = find(X(:, 6) == POS(1) & X(:, 7) == POS(2));
  Q = X(IDX, 10 : 17);
  V = max(Q, [], 2);
  H = plot(V);
  legend(["V(" int2str(POS) ")"]);
  title(["V(" int2str(POS) ")"]);
  grid on;
endfunction
