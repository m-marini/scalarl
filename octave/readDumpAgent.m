function [Q ACTIONS] = readDumpAgent(file, EPISODE = 1, W = 10, H = 10)
  N = W * H;
  QO = 4;
  NA = 8;
  X = csvread(file)(EPISODE, QO : QO + NA * N - 1);
  Q = permute(reshape(X, 8, W, H), [3 2 1]);
  [_ ACTIONS] = max(Q, [], 3);
  ACTIONS = ACTIONS - 1;
endfunction