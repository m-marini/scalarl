function [S Q] = readDump(file)
  X = csvread(file);
  N = size(X, 1);
  Q = reshape(X(:, 4 : end), N, 10, 10, 8);
  S = max(Q, 3);
endfunction
