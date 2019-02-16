
while true
do
    rm maze.model
    sbt run
    mv maze-kpis.csv data/maze-kpis-$(date +%Y%m%d%H%M%S).csv
done