#function plotAlphas()
  BASEPATH = "..";
  COL = 1;
  FNAMES = [
    BASEPATH "/data-lambda-0";
    BASEPATH "/data-lambda-5";
    BASEPATH "/data-base";
    BASEPATH "/data-lambda-8";
    BASEPATH "/data-lambda-9"
  ];
  LEGEND=[
    "0",
    "0.5",
    "0.7",
    "0.8",
    "0.9"
  ];
  plotStats(FNAMES, "lambda", LEGEND, 1);
