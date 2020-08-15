function analyzeEpisodes(X,              # The trace data
  NPOINTS = 300,  # The number of points in the charts
  LENGTH = 300    # The number of averaging samples in the charts
  )
  # Constants
  Landed = 1;
  OutOfFuel = 8;
  NSC = [Landed : OutOfFuel];

  StatusDescr = {
  "flying"
  "landed"
  "landed out of platform"
  "vertical crashed on platform"
  "vertical crash out of platform"
  "horizontal crash on platform"
  "horizontal crash out of platform"
  "out of range"
  "out of fuel"
  };
  
  Step = 2;
  StatusCode = 4;
  POSX = 5;
  POSY = 6;
  POSZ = 7;
  VX = 8;
  VY = 9;
  VZ = 10;
  Reward = 16;

  NC = 2;

  [X1 R] = episodes(X);
  EpisodeStep = X1(:, Step);
  F = frequency(X1);
  
  # Computes the StatusCode
  SCHIST = histc(X1(:, StatusCode), NSC);
  [DUMMY SORTX] = sort(SCHIST, "descend");
  SCHIST = SCHIST(SORTX);
  NSCHIST = NSC(SORTX);
  
  # Computes the distance from platform
  PF = -X1(:, POSX) - i * X1(:, POSY);
  [STEPB DP DPTREND DPMODE] = meanchart(abs(PF), NPOINTS, LENGTH);

  # Compute the after rewards
  [STEPB R RTREND RMODE] = meanchart(R, NPOINTS, LENGTH);
  
  NFS = sum(SCHIST >= 1);
  NR = max(3, NFS);
  
  # Draw charts
  clf;
  
  subplot(NR, NC, 1);
  autoplot(EpisodeStep(STEPB), [R RTREND]);
  grid on;
  title(sprintf("Episode final rewards\n%s Trend", RMODE));
  ylabel("Unit");
  xlabel("Step");

  subplot(NR, NC, 1 + NC);
  hist(X1(:, StatusCode), NSC);
  grid on;
  title(sprintf("Status distribution"));
  ylabel("# Episodes");
  xlabel("StatusCode");
  
  subplot(NR, NC, 1 + 2 * NC);
  autoplot(EpisodeStep(STEPB), [DP DPTREND]);
  grid on;
  title(sprintf("Platform distance\n%s Trend", DPMODE));
  ylabel("m");
  xlabel("Step");

  for I = 1 : length(NSCHIST)
    SC = NSCHIST(I);
    if SCHIST(I) >= 1
      Y = F{SC, 1};
      subplot(NR, NC, 2 + (I - 1) * NC);
      autoplot(Y(:, 1), Y(:, 2));
      grid on;
      title(sprintf("Frequency of %s", StatusDescr{SC + 1}));
      ylabel("1/episodes");
      xlabel("Step");
    endif
  endfor

  
  printf("Episodes\n");
  printf("%s rewards trend from %.1f to %.1f.\n", RMODE, RTREND(1), RTREND(end));
  printf("%s platform distance trend from %.0f m to %.0f m.\n", DPMODE, DPTREND(1), DPTREND(end));
  for I = 1 : length(NSCHIST)
    SC = NSCHIST(I);
    if SCHIST(I) > 0
      printf("%d cases of %s.\n", SCHIST(I), StatusDescr{SC+1});
    endif
  endfor
   
endfunction
