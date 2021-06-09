function Y = getEpisodes(
  X              # The trace datan the charts
  )
  END_IDX = find(X(:, 4) != 0);
  START_IDX = [1; END_IDX(1 : end -1) + 1];
  STATUS = X(END_IDX, 4);
  
  TOT_RETURN = zeros(size(START_IDX,1), 1);
  for I = 1 : size(TOT_RETURN, 1)
    TOT_RETURN(I) = sum(X([START_IDX(I) : END_IDX(I)], 16));
  endfor
  
  Y = [START_IDX END_IDX STATUS TOT_RETURN];
endfunction