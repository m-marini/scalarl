clear all;

function STATS = write_stats(file, data)
  STATS = [mean(data, 2) std(data, 0, 2) prctile(data, [0 5 25 50 75 95 100], 2)];
  csvwrite(file, STATS);
endfunction

basePath="../data/";
N = 100;
M = 1000;
ERRORS = zeros(M, N);
RETURNS = zeros(M, N);
STEPS = zeros(M, N);
for i = 1 : N
  file = [basePath "/maze-dump-" num2str(i) ".csv"];
  printf("Loading %s ...\n", file);
  X = csvread(file);
  STEPS(:, i) = X(:, 1);
  RETURNS(:, i) = X(:, 2);
  ERRORS(:, i) = X(:, 3);
endfor
STEPS = write_stats([basePath "/steps.csv"], STEPS);
RETURNS = write_stats([basePath "/returns.csv"], RETURNS);
ERRORS = write_stats([basePath "/errors.csv"], ERRORS);
