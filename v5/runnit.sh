#/bin/bash
# runnit.sh testid instanceId epochs
if [ -z "$3" ]; then
   n=99
else
   n=$3
fi
if [ -z "$2" ]; then
  id=1
else
  id=$2
fi
infile="lander-$1.yaml"
#sbt stage
rm stop
for ((i=0; i<$n; i++))
do
    if test -f "stop"; then
        echo 'Stopped due "stop" semphore file'
        exit 1;
    fi
    kpifile="kpi-$1-$id.csv"
    dumpfile="dump-$1-$id.csv"
    echo "../target/universal/stage/bin/scalarl --conf=$infile --epoch=$i --dumpFile=$dumpfile --kpiFile=$kpifile"
    ../target/universal/stage/bin/scalarl --conf=$infile --epoch=$i --dumpFile=$dumpfile --kpiFile=$kpifile
done
