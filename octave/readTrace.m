function [S0, Q0, ACTION, REWARD, S1, Q1, NQ0] = readTrace(file, W = 10, H = 10)
  # INPUT OFFSET: 1:200, 201:208,      209,      210,      211:410,           411:418,             419:426
  # INPUT OFFSET: 1:2*N, 2*N+1:2*N+NA, 2*N+NA+1, 2*N+NA+2, 2*N+NA+3:4*N+NA+2, 4*N+NA+3:4*N+2*NA+2, 4*N+2*NA+3:4*N+3*NA+2
  N = W * H;
  NA = 8;
  X = csvread(file);
  OS0 = 1;
  OQ0 = OS0 + 2 * N;
  OACTION = OQ0 + NA;
  OREWARD = OACTION + 1;
  OS1 = OREWARD + 1;
  OQ1 = OS1 + 2 * N;
  ONQ0 = OQ1 + NA;
  OEND = ONQ0 + NA;
  S0 = observation2Pos(X(:, OS0: OQ0 - 1));
  Q0 = X(:, OQ0 : OACTION - 1);
  ACTION = X(:, OACTION);
  REWARD = X(:, OREWARD);
  S1 = observation2Pos(X(:, OS1 : OQ1 - 1));
  Q1 = X(:, OQ1 : ONQ0 - 1);
  NQ0 = X(:, ONQ0 : OEND - 1);
endfunction