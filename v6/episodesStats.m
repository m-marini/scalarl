function episodesStats(
  X              # The trace data
  )
  
  IDX = getEpisodesIndices(X);
  N = size(IDX, 1);
 
  SC = [0 : 8];
  ST = X(IDX(:, 2), 4);
  H =  [SC' histc(ST, SC)];
  HI = find(H(:,2) > 0);
  H = H(HI, :);
  [_ HI] = sort(H(:,2), "descend");
  H = H(HI, :);
  M = size(H, 1);
  
  for I = 1 : M
    printf("%d cases of %s.\n", H(I, 2), statusDescr(H(I, 1)));
  endfor

endfunction