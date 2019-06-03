function [PREV_POS, POS, ACTIONS, REWARDS, PREV_Q, Q, PREV_Q1] = readTrace(file, episode=0, W = 10, H = 10)
  # INPUT         S0,    Q0,           ACTION,   REWARD,   S1,                Q1,                  NQ0
  # INPUT OFFSET: 1:200, 201:208,      209,      210,      211:410,           411:418,             419:426
  # INPUT OFFSET: 1:2*N, 2*N+1:2*N+NA, 2*N+NA+1, 2*N+NA+2, 2*N+NA+3:4*N+NA+2, 4*N+NA+3:4*N+2*NA+2, 4*N+2*NA+3:4*N+3*NA+2
  N = W * H;
  NA = 8;
  X = csvread(file);
  if (episode != 0)
    X = X(find(X(:, 1) == episode),:);
  endif
  PREV_POS = X(:,6 : 7);
  POS = X(:,8 : 9);
  ACTIONS = X(:, 3);
  REWARDS = X(:, 4);
  PREV_Q = X(:, 10 : 17);
  Q = X(:, 18 : 25);
  PREV_Q1 = X(:, 26 : 33);
endfunction
