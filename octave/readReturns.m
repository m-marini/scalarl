function R = readReturns(file)
  X = csvread(file);
  R = X(:, 2);
endfunction