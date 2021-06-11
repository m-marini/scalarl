function Y = extractPath(
  X,                        # The trace datan the charts
  STATUS_CODE,              # reference status code
  INDEX_CODE = 0,          # Index of status code episode (1 = first, 0 = last, -1 = seconf last)
  NO_PREVIOUS_EPISODE = 5,  # Number of previous episode
  NO_FUTHER_EPISODES = 0    # Number of further episode
  )
  
  EPS = getEpisodesIndices(X);
  N = size(EPS, 1);
  ST = X(EPS(:, 2), 4);
  REF_EPISODES = find(ST == STATUS_CODE);
  M = size(REF_EPISODES, 1);
  
  if INDEX_CODE > 0
     REF_EPISODE_IDX = min(INDEX_CODE, M);
  else
     REF_EPISODE_IDX = max(1, M + INDEX_CODE);
  endif
  REF_EPISODE = REF_EPISODES(REF_EPISODE_IDX);
  FROM = max(1, REF_EPISODE - NO_PREVIOUS_EPISODE);
  TO = min(REF_EPISODE + NO_FUTHER_EPISODES, N);

  for I = FROM : TO
    printf("Episode %d %s\n", I, statusDescr(ST(I)));
  endfor
  Y = X(EPS(FROM, 1) : EPS(TO, 2), [5 6 7 8 9 10 11 12 4]);
  
 endfunction