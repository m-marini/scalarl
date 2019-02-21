function [POS Q ACTION MAP] = readDumpAgent(file, EPISODE = 1, W = 10, H = 10)
  N = W * H;
  SO = 4;
  QO = SO + N * N * 2;
  NA = 8;
  X = csvread(file)(EPISODE, :);
  POS = zeros(N, 2);
  Q = zeros(N, NA);
  for i = 1 : N
    S = X(SO : SO + N - 1);
    P = observation2Pos(S);
    POS(i, :) = P;
    SO = SO + 2 * N;

    Q(i, :) = X(QO : QO + NA - 1);
    QO = QO + NA;
  endfor
  [_ ACTION] = max(Q');
  ACTION = ACTION' -1;
  MAP = zeros(H, W);
  for i = 1 : N
    MAP(POS(i, 1) + 1, POS(i, 2) + 1) = ACTION(i);
  endfor
endfunction