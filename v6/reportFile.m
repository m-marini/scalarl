function reportFile(FILE, LAST = 100, STRIDE=100)
  report(csvread(FILE), LAST, STRIDE);
endfunction
