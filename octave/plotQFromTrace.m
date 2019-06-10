function H = plotQFromTrace(file, POS, W = 10, H = 10)
  N = W * H;
  NA = 8;
  X = csvread(file);
  IDX = find(X(:, 6) == POS(1) & X(:, 7) == POS(2));
  Q = X(IDX, 10 : 17);
  H = plot(Q);
  legend(
   ["Q(" int2str(POS) ", 0)"],
   ["Q(" int2str(POS) ", 1)"],
   ["Q(" int2str(POS) ", 2)"],
   ["Q(" int2str(POS) ", 3)"],
   ["Q(" int2str(POS) ", 4)"],
   ["Q(" int2str(POS) ", 5)"],
   ["Q(" int2str(POS) ", 6)"],
   ["Q(" int2str(POS) ", 7)"]
   );
  title(["Q(" int2str(POS) ")"]);
  grid on;
endfunction
