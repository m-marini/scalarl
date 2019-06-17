#function plotAlphas()
  BASEPATH = "..";
  COL = 1;
  TITLE="alpha";
  PGN_PREFIX = "../analysis/alpha-";
  FNAMES = [
    BASEPATH "/data-alpha-10";
    BASEPATH "/data-alpha-30";
    BASEPATH "/data-base";
    BASEPATH "/data-alpha-300";
#    BASEPATH "/data-alpha-1"
  ];
  LEGEND=[
    "0.01",
    "0.03",
    "0.1",
    "0.3",
#    "1"
  ];
  plotStats(FNAMES, TITLE, LEGEND, COL, PGN_PREFIX);
