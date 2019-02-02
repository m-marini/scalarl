// Copyright (c) 2019 Marco Marini, marco.marini@mmarini.org
//
// Licensed under the MIT License (MIT);
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// https://opensource.org/licenses/MIT
//
// Permission is hereby granted, free of charge, to any person
// obtaining a copy of this software and associated documentation
// files (the "Software"), to deal in the Software without
// restriction, including without limitation the rights to use,
// copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the
// Software is furnished to do so, subject to the following
// conditions:
//
// The above copyright notice and this permission notice shall be
// included in all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
// EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
// OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
// NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
// HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
// WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
// OTHER DEALINGS IN THE SOFTWARE.

package org.mmarini.scalarl.envs

import scala.collection.Seq
import scala.util.Random
import org.mmarini.scalarl.Env
import org.mmarini.scalarl.Agent
import org.mmarini.scalarl.agents.QAgent
import org.mmarini.scalarl.Session
import org.mmarini.scalarl.agents.QAgentBuilder
import org.yaml.snakeyaml.Yaml
import java.io.FileReader
import scala.collection.JavaConverters._
import scala.collection.JavaConversions._
import org.deeplearning4j.util.ModelSerializer

object MazeMain {

  private def buildEnv(conf: Map[String, Any])(): Env = {
    val map = conf("env").asInstanceOf[java.util.Map[String, Any]].get("map").asInstanceOf[java.util.List[String]]
    MazeEnv.fromStrings(map)
  }

  private def buildAgent(conf: Map[String, Any])(): Agent = {
    val _agentConf = agentConf(conf)
    val numInputs = _agentConf("numInputs").asInstanceOf[Int]
    val numActions = _agentConf("numActions").asInstanceOf[Int]
    val numHiddens = _agentConf("numHiddens").asInstanceOf[Int]
    val seed = _agentConf.get("seed").map(_.asInstanceOf[Int])
    val epsilon = _agentConf("epsilon").asInstanceOf[Double]
    val gamma = _agentConf("gamma").asInstanceOf[Double]
    val learningRate = _agentConf("learningRate").asInstanceOf[Double]
    val builder1 = QAgentBuilder(numInputs, numActions).
      numHiddens1(numHiddens).
      numHiddens2(numHiddens).
      epsilon(epsilon).
      gamma(gamma).
      learningRate(learningRate)
    seed.map(s => builder1.seed(s)).getOrElse(builder1).build()
  }

  private def loadConfig(file: String): Map[String, Any] = {
    val conf = new Yaml().load(new FileReader(file))
    conf.asInstanceOf[java.util.Map[String, Any]].toMap
  }

  private def agentConf(conf: Map[String, Any]): Map[String, Any] = conf("agent").asInstanceOf[java.util.Map[String, Any]].toMap

  def main(args: Array[String]) {
    val conf = loadConfig(if (args.isEmpty) "maze.yaml" else args(0))
    val sessionConf = conf("session").asInstanceOf[java.util.Map[String, Any]].toMap
    val numEpisodes = sessionConf("numEpisodes").asInstanceOf[Int]
    val sync = sessionConf("sync").asInstanceOf[Int]
    val mode = sessionConf.get("mode").map(_.toString).getOrElse("human")
    val _agentConf = agentConf(conf)
    val model = _agentConf("model").toString
    val session = new Session(numEpisodes, buildEnv(conf), buildAgent(conf), sync = sync, mode = mode).run()
    session.agent.asInstanceOf[QAgent].writeModel(model)
  }
}
