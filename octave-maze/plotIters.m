#function plotAlphas()
  BASEPATH = "..";
  COL = 1;
  TITLE="Batch iteations";
  PGN_PREFIX = "../analysis/iter-";
  FNAMES = [
    BASEPATH "/data-base";
    BASEPATH "/data-iter-3";
    BASEPATH "/data-iter-10";
    BASEPATH "/data-iter-30";
    BASEPATH "/data-iter-100"
  ];
  LEGEND=[
    "1",
    "3",
    "10",
    "30",
    "100"
  ];
  plotStats(FNAMES, TITLE, LEGEND, COL, PGN_PREFIX);
