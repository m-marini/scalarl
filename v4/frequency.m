function P = frequency(X)
  # Constants
  Step = 2;
  StatusCode = 4;
  SCodes = [1 : 8];

  P = {};
  for SC = SCodes
    H = [0 0];
    J = 2;
    COUNTER = 1;
    for I = 1 : size(X, 1)
      if X(I, StatusCode) == SC
        H(J, 1) = X(I, Step);
        H(J, 2) = 1 / COUNTER;
        COUNTER = 1;
        J = J + 1;
      else
        COUNTER = COUNTER + 1;
      endif
    endfor
    P{SC, 1} = H;
  endfor
endfunction
