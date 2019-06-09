function V = vFromEpisode(X)
## -*- texinfo -*-
## @deftypefn  {Function File} @var{Q} = vFromEpisode (@var{X})
## Return the state values from episode dump
## 
## @var{X} is vector with the episode dump
##
## The return values @var{V} is a matrix containing the status values at the end
## of episode in the form 10 x 10
## @end deftypefn
  [Q MASK] = qFromEpisode(X);
  V = zeros(1, 100);
  for i = 1 : 100
    V(i) = max(Q(i, find(MASK(i, :))));
  endfor
  V = reshape(V, 10, 10);
endfunction
