rm lander-samples.csv
sbt "runMain org.mmarini.scalarl.ts.envs.Main trace-lander-0.yaml"
sbt "runMain org.mmarini.scalarl.ts.envs.Main trace-lander-1.yaml"
sbt "runMain org.mmarini.scalarl.ts.envs.Main trace-lander-2.yaml"
sbt "runMain org.mmarini.scalarl.ts.envs.Main trace-lander-3.yaml"
sbt "runMain org.mmarini.scalarl.ts.envs.Main trace-lander-4.yaml"
sbt "runMain org.mmarini.scalarl.ts.envs.Main trace-lander-5.yaml"
cp lander.zip lander-1.0.0.zip
