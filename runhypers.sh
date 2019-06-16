#/bin/bash
SOURCE=hypers/*.yaml
for f in $SOURCE
do
  NAME=$(expr match "$f" 'hypers/\(.*\).yaml')
  DATAPATH=data-$NAME
  if [[ -d "$DATAPATH" ]]
  then
    echo Skipping $f
  else
    mkdir -p data
    rm -f data/*
    cp $f data/maze.yaml
    cp $f maze.yaml
    echo Runnig on $f
    ./runnit.sh
    mv data $DATAPATH
  fi
done