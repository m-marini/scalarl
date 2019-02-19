function [V0 V1 EXPECTED ERRORS] = qlearn(Q0, Q1, ACTION, REWARD, GAMMA)
  [_ GREEDY] = max(Q0');
  V0 = max(Q0')';
  V1 = max(Q1')';
  V1(REWARD == 1) = 0;
  DELTA = REWARD + GAMMA .* V1 - V0;
  DQ0 = zeros(size(Q0));
  for i = 1 : size(Q0, 1)
    DQ0(i , ACTION(i) + 1) = DELTA(i);
  endfor
  EXPECTED = Q0 + DELTA;
  ERRORS = sum((EXPECTED - Q0).^2, 2);
endfunction