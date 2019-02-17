function Q = readQ(file, EPISODE = 1, W = 10, H = 10)
  N = W * H;
  NA = 8;
  OFFSET = 4 + N * N * 2;
  X = csvread(file);
  A = [];
  IDX = [0: NA : (N - 1) * NA] + OFFSET;
  Q = [];
  for i = IDX
    Q = [Q ; X(EPISODE, i : i + NA -1)];
  endfor
endfunction