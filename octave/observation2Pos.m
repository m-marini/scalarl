function POS = observation2Pos(X, W = 10, H = 10)
  N = W * H;
  M = size(X, 1);
  IDX = zeros(M, 1);
  for i = 1 : M
    IDX(i) = find(X(i, 1 : N)) - 1;
  endfor
  POS = [ mod(IDX, W) floor(IDX / W)];
endfunction