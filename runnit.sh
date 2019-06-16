#/bin/bash
rm maze-dump.csv
for i in {1..100}
do
    rm maze.zip
    sbt run
    mv maze-dump.csv data/maze-dump-$i.csv
done
