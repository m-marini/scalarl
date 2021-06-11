function DESCR = statusDescr(
  X # status codes
  )
  
  Descriptions = {
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

  DESCR = Descriptions{X + 1};
  
endfunction
