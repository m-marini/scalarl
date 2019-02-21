function R = readStepCount(file)
  X = csvread(file);
  R = X(:, 1);
endfunction