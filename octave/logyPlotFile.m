function H = logyPlotFile(filename)
  Y = csvread(filename);
  H = semilogy(Y(:, [8, 6, 4, 1]));
  title(["File " filename]);
  legend("95%", "50%", "5%", "Avg");
  grid on;
endfunction
