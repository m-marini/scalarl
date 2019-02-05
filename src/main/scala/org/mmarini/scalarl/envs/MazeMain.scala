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
import java.io.FileWriter
import org.mmarini.scalarl.AgentKpi
import org.mmarini.scalarl.DefaultAgentKpi

object MazeMain {

  private def buildEnv(conf: Configuration): Env = {
    val map = conf.getConf("env").getList[String]("map")
    MazeEnv.fromStrings(map)
  }

  private def buildAgent(conf: Configuration): Agent = {
    val numInputs = conf.getConf("agent").getInt("numInputs").get
    val numActions = conf.getConf("agent").getInt("numActions").get
    val numHiddens = conf.getConf("agent").getInt("numHiddens").get
    val seed = conf.getConf("agent").getLong("seed").get
    val epsilon = conf.getConf("agent").getDouble("epsilon").get
    val gamma = conf.getConf("agent").getDouble("gamma").get
    val learningRate = conf.getConf("agent").getDouble("learningRate").get
    val maxAbsGrads = conf.getConf("agent").getDouble("maxAbsGradients").get
    val maxAbsParams = conf.getConf("agent").getDouble("maxAbsParameters").get
    val model = conf.getConf("agent").getString("model").get
    QAgentBuilder(numInputs, numActions).
      numHiddens1(numHiddens).
      numHiddens2(numHiddens).
      epsilon(epsilon).
      gamma(gamma).
      learningRate(learningRate).
      maxAbsGradient(maxAbsGrads).
      maxAbsParams(maxAbsParams).
      seed(seed).
      file(model).
      build()
  }

  private def loadConfig(file: String): Configuration = {
    val conf = new Yaml().load(new FileReader(file))
    new Configuration(conf.asInstanceOf[java.util.Map[String, Any]].toMap)
  }

  private def agentConf(conf: Map[String, Any]) = conf("agent").asInstanceOf[java.util.Map[String, Any]].toMap
  private def sessionConf(conf: Map[String, Any]) = conf("session").asInstanceOf[java.util.Map[String, Any]].toMap

  private def writerCvsHeader(file: String)(header: Seq[String]) {
    val fw = new FileWriter(file)
    fw.write(header.mkString(",") + "\n")
    fw.close();
  }
  private def writeCvsRecord(file: String)(kpi: AgentKpi) {
    val fw = new FileWriter(file, true)
    val record = Seq(
      kpi.asInstanceOf[DefaultAgentKpi].episodeCount.toString,
      kpi.stepCount().toString,
      kpi.returnValue().toString,
      kpi.avgLoss().toString)
    fw.write(record.mkString(",") + "\n")
    fw.close();
  }

  def main(args: Array[String]) {
    val conf = loadConfig(if (args.isEmpty) "maze.yaml" else args(0))
    val numEpisodes = conf.getConf("session").getInt("numEpisodes").get
    val sync = conf.getConf("session").getLong("sync").get
    val mode = conf.getConf("session").getString("mode").get
    val model = conf.getConf("agent").getString("model").get
    val kpis = conf.getConf("agent").getString("kpis").get
    writerCvsHeader(kpis)(Seq("Episode", "StepCount", "ReturnValue", "AvgLoss"))
    val session = new Session(
      numEpisodes,
      buildEnv(conf),
      buildAgent(conf),
      sync = sync,
      mode = mode,
      onEpisode = Some((session) => {
        session.agent.asInstanceOf[QAgent].writeModel(model)
        None.orNull
      }))
    session.agent.agentKpiObs.subscribe(writeCvsRecord(kpis)_)
    session.run()
  }
}

class Configuration(conf: Map[String, Any]) {
  def getConf(key: String): Configuration = conf.get(key) match {
    case Some(m: java.util.Map[String, Any]) => new Configuration(m.toMap)
    case _                                   => new Configuration(Map())
  }

  def getNumber(key: String): Option[Number] = conf.get(key) match {
    case Some(n: Number) => Some(n)
    case _               => None
  }

  def getInt(key: String): Option[Int] = getNumber(key).map(_.intValue())

  def getLong(key: String): Option[Long] = getNumber(key).map(_.longValue())

  def getDouble(key: String): Option[Double] = getNumber(key).map(_.doubleValue())

  def getString(key: String): Option[String] = conf.get(key) match {
    case Some(s: String) => Some(s)
    case _               => None
  }

  def getList[T](key: String): List[T] = {
    val x = conf.get(key)
    x match {
      case Some(l: java.util.List[T]) => l.asInstanceOf[java.util.List[T]].toList
      case _                          => List()
    }
  }
}
