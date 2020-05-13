function model(X, T = 500)
  
  # Number of steps
  n = size(X, 1);
  # Number of charts
  NC = 2;

  XX = X(:, 9 : 10);
  
  [Y XN] = [slieAvg(XX, T, 1)];
  
  subplot(1,1,1);
  plot(XN, Y);
  grid on;
  grid minor on;
  title(sprintf("Average model size on %d samples", T));
  ylabel("# Samples");
  ylabel("Steps");
  legend("Model size", "Queue size");
  
endfunction
