function Y=softmax(X)
  Z = exp(X);
  T = sum(Z, 2);
  Y = Z ./ T;
endfunction
