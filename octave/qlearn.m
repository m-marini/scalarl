function [ERRORS ESTIMATION V0 V1 EXPECTED S0, Q0, ACTION, REWARD, S1, Q1, NQ0] = qlearn(file, GAMMA = 0.999, W = 10, H = 10)
  [S0, Q0, ACTION, REWARD, S1, Q1, NQ0] = readTrace(file, W, H);
  [_ GREEDY] = max(Q0');
  V0 = max(Q0')';
  V1 = max(Q1')';
  V1(REWARD == 1) = 0;
  DELTA = REWARD + GAMMA .* V1 - V0;
  DQ0 = zeros(size(Q0));
  for i = 1 : size(Q0, 1)
    DQ0(i , ACTION(i) + 1) = DELTA(i);
  endfor
  EXPECTED = Q0 + DQ0;
  ESTIMATION = sum((EXPECTED - Q0).^2, 2);
  ERRORS = sum((EXPECTED - NQ0).^2, 2); 
endfunction