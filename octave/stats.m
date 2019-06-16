function [RETURNS ERROS] = stats(fname, N = 100)
## -*- texinfo -*-
## @deftypefn  {Function File} [@var(RETURNS) @var(ERRORS)] = stats(@var{fname}, @var(N))
## Read the dump file from the data folder
## computes the statistic on episodes and
## write the data to returns.csv file and errors.csv file each the column is
## contains:
##
##  - [:, 1] MEAN
##
##  - [:, 2] STD
##
##  - [:, 3] 0 percentile (MIN)
##
##  - [:, 4] 5 percentile
##
##  - [:, 5] 25 percentile
##
##  - [:, 6] 50 percentile (MEDIAN)
##
##  - [:, 7] 75 percentile
##
##  - [:, 8] 95 percentile
##
##  - [:, 9] 100 percentile (MAX)
##
## @var{fname} the folder containing the dump files
##
## @var{N} the number of dump files
##
## The return values @var{RETURNS} is a matrix containing the returns statistics
##
## The return values @var{ERRORS} is a matrix containing the errors statistics
## @end deftypefn
 N = 100;
 M = 0;
 for i = 1 : N
  file = [fname "/maze-dump-" num2str(i) ".csv"];
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
 printf("Writing %s ...\n", [fname "/returns.csv"]);
 RETURNS = write_stats([fname "/returns.csv"], RETURNS);
 printf("Writing %s ...\n", [fname "/errors.csv"]);
 ERRORS = write_stats([fname "/errors.csv"], ERRORS);
endfunction
 
function STATS = write_stats(file, data)
  STATS = [mean(data, 2) std(data, 0, 2) prctile(data, [0 5 25 50 75 95 100], 2)];
  csvwrite(file, STATS);
endfunction
