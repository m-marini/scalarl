#/bin/bash
n=99 #$2
#sbt stage
for i in {0..99}
do
    ./target/universal/stage/bin/scalarl $1 $i
    #./target/universal/stage/bin/org_mmarini_scalarl_v-3_envs_main $1 $i
    #sbt "runMain org.mmarini.scalarl.v1.envs.Main $f $i"
done
