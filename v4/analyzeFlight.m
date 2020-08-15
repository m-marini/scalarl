function analyzeFlight(
  X,              # The trace data
  NPOINTS = 300,  # The number of points in the charts
  LENGTH = 300    # The number of averaging samples in the charts
  )
  # Constants
  Flying = 0;

  Step = 2;
  StatusCode = 4;
  POSX = 5;
  POSY = 6;
  POSZ = 7;
  VX = 8;
  VY = 9;
  VZ = 10;
  Reward = 16;

  NR = 4;
  NC = 1;

  # Extracts the status codes
  STATUS = X(:, StatusCode);

  # Extract the flying step indices
  FLYX = find(STATUS == Flying);
  if FLYX(1) == 1
    # Bypass initial step
    FLYX = FLYX(2 : end, :);
  endif
  
  # Filter flying status
  X1 = X(FLYX, :);
  EpisodeStep = X1(:, Step);
  N = size(X1, 1);
  
  # Computes the flying h speed over time
  V = X1(:, VX) + i * X1(:, VY);
  VHF = abs(V);
  
  # Compute the flying direction error over the time
  PF = -X1(:, POSX) - i * X1(:, POSY); 
  MERHO = PF .* V;
  for a = 1 : N
    if VHF(a) > 0
      MERHO(a) = MERHO(a) / VHF(a);
    else
      MERHO(a) = -1;
    endif
  endfor
  MERHO = rad2deg(abs(arg(MERHO)));
  [STEPF MERHO MERHOTREND MERHOMODE] = meanchart(MERHO, NPOINTS, LENGTH);

  # Computes the flying z speed over time
  [STEPF VZF VZFTREND VZFMODE] = meanchart(X1(:, VZ), NPOINTS, LENGTH);
 
  # Compute the flying rewards
  [STEPF RWF RWFTREND RWFMODE] = meanchart(X(FLYX - 1, Reward), NPOINTS, LENGTH);
  
  # Computes the flying h speed over time
  [STEPF VHF VHFTREND VHFMODE] = meanchart(VHF, NPOINTS, LENGTH);

  # Draw charts
  clf;
  subplot(NR, NC, 1);
  autoplot(EpisodeStep(STEPF), [RWF RWFTREND]);
  grid on;
  title(sprintf("Average Flying rewards\n%s Trend", RWFMODE));
  ylabel("Unit");
  xlabel("Step");

  subplot(NR, NC, 1 + NC);
  autoplot(EpisodeStep(STEPF), [MERHO MERHOTREND]);
  grid on;
  title(sprintf("Average Direction error\n%s Trend", MERHOMODE));
  ylabel("DEG");
  xlabel("Step");

  subplot(NR, NC, 1 + 2 * NC);
  autoplot(EpisodeStep(STEPF), [VHF VHFTREND]);
  grid on;
  title(sprintf("Average H speed\n%s Trend", VHFMODE));
  ylabel("m/s");
  xlabel("Step");
  
  subplot(NR, NC, 1 + 3 * NC);
  autoplot(EpisodeStep(STEPF), [VZF VZFTREND]);
  grid on;
  title(sprintf("Average Z speed\n%s Trend", VZFMODE));
  ylabel("m/s");
  xlabel("Step");

  printf("Flying status\n");
  printf("%s rewards trend from %.1f to %.1f.\n", RWFMODE, RWFTREND(1), RWFTREND(end));
  printf("%s direction error trend from %.0f DEG to %.0f DEG.\n", MERHOMODE, MERHOTREND(1), MERHOTREND(end));
  printf("%s horizontal speed trend from %.1f m/s to %.1f m/s.\n", VHFMODE, VHFTREND(1), VHFTREND(end));
  printf("%s vertical speed trend from %.1f m/s to %.1f m/s.\n", VZFMODE, VZFTREND(1), VZFTREND(end));
   
endfunction
