function [Y1 P MODE] = regression(Y)
  [YLIN PLIN] = linearRegression(Y);
  [YLOG PLOG] = logRegression(Y);
  ELIN = sum((Y - YLIN) .^2);
  ELOG = sum((Y - YLOG) .^2);
  if ELIN <= ELOG
    Y1 = YLIN;
    P = PLIN;
    E = ELIN;
    MODE = "Linear";
  else
    Y1 = YLOG;
    P = PLOG;
    E = ELOG;
    MODE = "Logarithmic";
  endif
  if min(Y) > 0
    [YEXP PEXP] = expRegression(Y);
    EEXP = sum((Y - YEXP) .^2);
    if EEXP < E
      Y1 = YEXP;
      P = PEXP;
      MODE = "Exponential";
    endif
  endif
  if P(1) > 0
    MODE = [MODE " Increasing"];
  elseif P(1) < 0
    MODE = [MODE " Decreasing"];
  else
    MODE = "Constant";
  endif
endfunction
