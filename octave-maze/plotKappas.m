#function plotAlphas()
  BASEPATH = "..";
  COL = 1;
  TITLE="kappa";
  PGN_PREFIX = "../analysis/";
  FNAMES = [
    BASEPATH "/data-base";
    BASEPATH "/data-kappa-2";
    BASEPATH "/data-kappa-4";
    BASEPATH "/data-kappa-8";
    BASEPATH "/data-kappa-16"
  ];
  LEGEND=[
    "1",
    "2",
    "4",
    "8",
    "16"
  ];
  plotStats(FNAMES, TITLE, LEGEND, COL, PGN_PREFIX);
