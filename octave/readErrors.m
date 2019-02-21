function E = readErrors(file)
  X = csvread(file);
  E = X(:, 3);
endfunction