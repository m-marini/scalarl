# Read the dump files from ../data folder,
# the statistics ara truncated to the minimun number of episodes
# compute the statistic on episodes and
# write the statistic file

clear all;
basePath="../data/";
N = 100;

function STATS = write_stats(file, data)
  STATS = [mean(data, 2) std(data, 0, 2) prctile(data, [0 5 25 50 75 95 100], 2)];
  csvwrite(file, STATS);
endfunction

M = 0;
for i = 1 : N
  file = [basePath "/maze-dump-" num2str(i) ".csv"];
  printf("Loading %s ...\n", file);
  X = csvread(file);
  L = size(X, 1);
  if i == 1
    RETURNS(:, i) = X(:, 2);
    ERRORS(:, i) = X(:, 3);
    M = L;
  elseif L < M
    RETURNS = RETURNS(1 : L, :);
    ERRORS = ERRORS(1 : L, :);
    RETURNS(:, i) = X(:, 2);
    ERRORS(:, i) = X(:, 3);
    M = L;
  elseif L > M
    X = X(1 : M, :);
    RETURNS(:, i) = X(:, 2);
    ERRORS(:, i) = X(:, 3);
  else
    RETURNS(:, i) = X(:, 2);
    ERRORS(:, i) = X(:, 3);
  endif
endfor
M
RETURNS = write_stats([basePath "/returns.csv"], RETURNS);
ERRORS = write_stats([basePath "/errors.csv"], ERRORS);
