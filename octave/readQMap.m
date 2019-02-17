function MAP = readQMap(file, EPISODE = 1, W = 10, H = 10)
  X = readQDump(file, EPISODE, W, H);
  [_ AC] = max(X(:, 3 : end)');
  MAP = AC';
  endfunction