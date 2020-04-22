#/bin/bash
n=99 #$2
#sbt stage
for i in {0..99}
do
    ./target/universal/stage/bin/org_mmarini_scalarl_v-2_envs_main $f $i
    #sbt "runMain org.mmarini.scalarl.v2.envs.Main $f $i"
done
