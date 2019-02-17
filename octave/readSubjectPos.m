function POS = readSubjectPos(file, EPISODE = 1, W = 10, H = 10)
  N = W * H;
  OFFSET = 4;
  X = csvread(file);
  A = [];
  IDX = [0: N * 2 : 2 * N * (N - 1)] + OFFSET + N;
  for i = IDX
    S = X(EPISODE, i : i + N);
    A = [A; (find(S, 1) - 1)];
  endfor
  POS = [ mod(A, W) floor(A / W)];
endfunction