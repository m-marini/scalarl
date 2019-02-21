
while true
do
    rm maze.zip
    sbt run
    mv maze-dump.csv data/maze-dump-$(date +%Y%m%d%H%M%S).csv
done