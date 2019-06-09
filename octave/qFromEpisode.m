function [Q MASK] = qFromEpisode(X)
## -*- texinfo -*-
## @deftypefn  {Function File} [@var{Q} @var{MASK}] = qFromEpisode(@var{X})
## Return the Q values from episode dump
## 
## @var{X} is vector with the episode dump
##
## The return values
##
## @var{Q} is a tensor of the Q values at the end of episode
## with the shape 100, 8
##
## @var{MASK} is vector with the available actins flags
## @end deftypefn
  Q = reshape(X(1, 4 : 803), 8, 100)';
  MASK = reshape(X(1, 804 : 1603), 8, 100)';
endfunction
