function [Y X] = averageFile(FILEIN, FILEOUT)
  X = csvread(FILEIN);
  Y = averageDump(X);
  csvwrite(FILEOUT, Y);
endfunction
