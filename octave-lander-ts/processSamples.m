function Y = processSamples(X, GAMMA = 0.999)
## -*- texinfo -*-
## @deftypefn  {Function File} [@var{SAMPLES} var@{LABELS}] = processSamples (@var{X})
## Return the learning samples with related labels
## 
## @var{X} is the trace data
####
## The return values @var{Y} is a matrix containing the learning case
## @end deftypefn
  # Reverse
  Y = flipud(X);
  # clear last uncompleted episode
  ENDIDX = 38;
  ENDEPISODES = find(Y(:, ENDIDX) != 0);
  Y = Y(ENDEPISODES(1) : end, :);
  N = size(Y, 1);
  RET = 0;
  G = 1;
  # for each episode computes the action values
  # remove the end status from episodes (not significant for learning)
  for i = 1 : N
    if Y(i, ENDIDX)
      RET = 0;
      G = GAMMA;
    endif
    G = G * GAMMA;
    RET = RET * G + Y(i, 37);
    IDX = find(Y(i, 37: 51));
    Y(i, IDX + 21) = RET;
  endfor

  # reverse and split
  Y = flipud(Y)(:, 1 : 21 + 15);
endfunction
