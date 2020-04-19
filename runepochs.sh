#/bin/bash
f="v1/lander-tiles-es-0.yaml" #$1
n=99 #$2
#sbt stage
for i in {0..99}
do
    ./target/universal/stage/bin/org_mmarini_scalarl_v-1_envs_main $f $i
    #sbt "runMain org.mmarini.scalarl.v1.envs.Main $f $i"
done
