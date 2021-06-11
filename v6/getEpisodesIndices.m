function Y = getEpisodesIndices(
  X              # The trace datan the charts
  )
  END_IDX = find(X(:, 4) != 0);
  START_IDX = [1; END_IDX(1 : end -1) + 1];
  Y = [START_IDX END_IDX];
  
endfunction