function POS = observation2Pos(X, W = 10, H = 10)
  N = W * H;
  M = size(X);
  [_ IDX] = find(X(:, N + 1: 2 * N));
  IDX = IDX - 1;
  POS = [ mod(IDX, W) floor(IDX / W)];
endfunction