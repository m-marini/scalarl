function [AVGSTEP AVGRET AVGERR] = process_csv()
  [STEPS RETS ERRS] = loadFolder("data/*.csv");
  AVGSTEP = mean(STEPS, 2);
  AVGRET = mean(RETS, 2);
  AVGERR = mean(ERRS, 2);
endfunction

function [STEPS RETS ERRS] = loadFolder(FOLDER)
  LS=ls(["-1 "  FOLDER]);
  n = size(LS, 1);
  STEPS = [];
  RETS = [];
  ERRS = [];
  for i = 1 : n
    FILE = LS(i, :);
    printf("Loading %s ...\n", FILE)
    [S R E] = loadFile(FILE);
    STEPS = [STEPS S];
    RETS = [RETS R];
    ERRS = [ERRS E];
  endfor
endfunction

function [STEP RET ERR] = loadFile(file)
  R = csvread(file);
  STEP = R(:, 2);
  RET = R(:, 3);
  ERR = R(:, 4);
endfunction