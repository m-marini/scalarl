function [Y R] = episodes(X)
  # Constants
  Flying = 0;

  StatusCode = 4;
  Reward = 16;

  # Extract the final step of episode (not flying)
  ENDX = find(X(:, StatusCode) != Flying);
  if ENDX(1) == 1
    # Bypass initial step
    ENDX = ENDX(2 : end, :);
  endif
  Y = X(ENDX, :);
  R = X(ENDX - 1, Reward);
endfunction
