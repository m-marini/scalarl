function Y = extractPath(
  X,                        # The trace datan the charts
  STATUS_CODE,              # reference status code
  INDEX_CODE = -1,          # Index of status code episode (-1 = last)
  NO_PREVIOUS_EPISODE = 5,  # Number of previous episode
  NO_FUTHER_EPISODES = 0    # Number of further episode
  )
  N = size(EPISODES, 1);
  IDX = [];
  for I = 1 : size(EPISODES, 1)
    IDX = [IDX; [EPISODES(I, 1) : EPISODES(I, 2)]'];
  endfor
  Y = X(IDX, 5 : 7);
 endfunction