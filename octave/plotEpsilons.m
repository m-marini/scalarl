#function plotAlphas()
  BASEPATH = "..";
  COL = 1;
  FNAMES = [
    BASEPATH "/data-epsilon-1";
    BASEPATH "/data-epsilon-3";
    BASEPATH "/data-base";
    BASEPATH "/data-epsilon-30";
    BASEPATH "/data-epsilon-100"
  ];
  LEGEND=[
    "0.001",
    "0.003",
    "0.01",
    "0.03",
    "0.1"
  ];
  plotStats(FNAMES, "epsilon", LEGEND, 1);
