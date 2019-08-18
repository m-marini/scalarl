#function plotAlphas()
  BASEPATH = "..";
  COL = 1;
  TITLE="History";
  PGN_PREFIX = "../analysis/hist-";
  FNAMES = [
    BASEPATH "/data-hist-1";
    BASEPATH "/data-hist-10";
    BASEPATH "/data-base";
    BASEPATH "/data-hist-300";
    BASEPATH "/data-hist-1000"
  ];
  LEGEND=[
    "1",
    "10",
    "100",
    "300",
    "1000"
  ];
  plotStats(FNAMES, TITLE, LEGEND, COL, PGN_PREFIX);
