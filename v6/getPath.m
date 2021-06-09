function Y = getPath(
  X,              # The trace datan the charts
  EPISODES        # Episodes data
  )
  N = size(EPISODES, 1);
  IDX = [];
  for I = 1 : size(EPISODES, 1)
    IDX = [IDX; [EPISODES(I, 1) : EPISODES(I, 2)]'];
  endfor
  Y = X(IDX, 5 : 7);
 endfunction