function QDUMP = readQDump(file, EPISODE = 1, W = 10, H = 10)
  P = readSubjectPos(file, EPISODE, W, H);
  Q = readQ(file, EPISODE, W, H);
  QDUMP = [P Q];
endfunction