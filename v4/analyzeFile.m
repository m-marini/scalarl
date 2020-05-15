function analyzeFile(FILE, LAST = 100)
  analyze(csvread(FILE), LAST);
endfunction
