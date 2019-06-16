#/bin/bash
rm maze-dump.csv
for i in {1..100}
do
    rm maze.zip
    target/universal/stage/bin/scalarl
    mv maze-dump.csv data/maze-dump-$i.csv
done
