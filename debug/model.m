function model(X, T=500)
  
  # Number of steps
  n = size(X, 1);
  # Number of charts
  NC = 2;

  XX = X(:, 11:12);
  
  Y = [movingAvg(XX, T)];
  
  subplot(1,1,1);
  plot(Y);
  grid on;
  grid minor on;
  title(sprintf("Average model size on %d samples", T));
  ylabel("# Samples");
  legend("Model size", "Queue size");
  
endfunction
