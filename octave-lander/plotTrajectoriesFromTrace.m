function [L RX RY RZ] = plotTrajectoriesFromTrace(X)
## -*- texinfo -*-
## @deftypefn  {Function File} [@var{R} @var[V]] = plotTrajectoriesFromTrace(@var{X})
## Return the trajectory from trace data
## 
## @var{X} is the trace data
####
## The return values @var{L} is a vector containg the length of trajectories
## The return values @var{RX} is a matrix containing the X coordinate of trajectories
## The return values @var{RY} is a matrix containing the Y coordinate of trajectories
## The return values @var{RZ} is a matrix containing the Z coordinate of trajectories
## @end deftypefn
 R = statusFromTrace(X);
 EU = endUpFromTrace(X);
 EUIDX = find(EU);

 # Number of trajectories
 N = size(EUIDX, 1) + 1;
 
 L=zeros(N, 1);
 K = 1;
 for i = 1 : N - 1
   L(i) = EUIDX(i) - K + 1;
   K = EUIDX(i) + 1;
 endfor
 L(end) = size(X, 1) - K + 1;
 M = max(L);
 RX = M;
 
 RX = zeros(M, N);
 RY = zeros(M, N);
 RZ = zeros(M, N);
 
 FROM = 1;
 for i = 1 : N
   LT = L(i);
   TO = FROM + LT - 1;
   RX(1 : LT, i) = R(FROM : TO, 1);
   RY(1 : LT, i) = R(FROM : TO, 2);
   RZ(1 : LT, i) = R(FROM : TO, 3);
   if LT < M
     ## Pad remaining data
     RX(LT + 1 : M, i) = R(TO, 1);
     RY(LT + 1 : M, i) = R(TO, 2);
     RZ(LT + 1 : M, i) = R(TO, 3);
   endif
   FROM = TO + 1;
 endfor
 
 plot3(RX, RY, RZ);
 grid on ;
endfunction
