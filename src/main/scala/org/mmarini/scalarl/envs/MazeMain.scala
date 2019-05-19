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

import java.io.FileReader
import java.io.Writer

import scala.collection.JavaConversions.`deprecated asScalaBuffer`
import scala.collection.JavaConversions.`deprecated mapAsScalaMap`
import scala.collection.JavaConversions.`deprecated seqAsJavaList`
import scala.collection.Seq

import org.mmarini.scalarl.Agent
import org.mmarini.scalarl.Env
import org.mmarini.scalarl.FileUtils.withFile
import org.mmarini.scalarl.FileUtils.writeINDArray
import org.mmarini.scalarl.Session
import org.mmarini.scalarl.agents.QAgent
import org.mmarini.scalarl.agents.QAgentBuilder
import org.nd4j.linalg.factory.Nd4j
import org.yaml.snakeyaml.Yaml
import org.nd4j.linalg.api.ndarray.INDArray
import org.mmarini.scalarl.Episode
import com.typesafe.scalalogging.LazyLogging
import org.mmarini.scalarl.Step
import org.mmarini.scalarl.agents.TDQAgent

object MazeMain extends LazyLogging {
  private def buildEnv(conf: Configuration): Env = {
    val map = conf.getConf("env").getList[String]("map")
    MazeEnv.fromStrings(map)
  }

  private def buildAgent(conf: Configuration): Agent = {
    val numInputs = conf.getConf("agent").getInt("numInputs").get
    val numActions = conf.getConf("agent").getInt("numActions").get
    val numHiddens = conf.getConf("agent").getList[Int]("numHiddens")
    val seed = conf.getConf("agent").getLong("seed").get
    val epsilon = conf.getConf("agent").getDouble("epsilon").get
    val gamma = conf.getConf("agent").getDouble("gamma").get
    val learningRate = conf.getConf("agent").getDouble("learningRate").get
    val maxAbsGrads = conf.getConf("agent").getDouble("maxAbsGradients").get
    val maxAbsParams = conf.getConf("agent").getDouble("maxAbsParameters").get
    val model = conf.getConf("agent").getString("model").get
    QAgentBuilder(numInputs, numActions).
      numHiddens(numHiddens.toArray).
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

  /**
   * Returns the dump data array of the episode
   * The data array is composed by:
   *
   * - stepCount
   * - returnValue
   * - average loss
   * - 10 x 10 x 8 of q action values for each state for each action
   */
  private def createDump(episode: Episode): INDArray = {
    val session = episode.session
    val qagent = episode.agent.asInstanceOf[QAgent]
    val kpi = Nd4j.create(Array(Array(episode.stepCount, episode.returnValue, episode.avgLoss)))
    val states = episode.env.asInstanceOf[MazeEnv].dumpStates
    val q = states.map(qagent.q)
    val qMat = Nd4j.vstack(q).ravel()
    Nd4j.hstack(kpi, qMat)
  }

  /**
   * Returns the trace data array of the step
   * The data array is composed by:
   *
   * - episodeCount
   * - stepCount
   * - action
   * - reward
   * - endUp flag
   * - prev row position
   * - prev col position
   * - result row position
   * - result col position
   * - prev q
   * - result q
   * - prev q1
   */
  private def createTrace(step: Step): INDArray = {
    val prevEnv = step.prevEnv.asInstanceOf[MazeEnv]
    val prevPos = prevEnv.subject
    val env = step.env.asInstanceOf[MazeEnv]
    val pos = env.subject
    val head = Nd4j.create(Array(Array(
      step.episode.toDouble,
      step.step.toDouble,
      step.action.toDouble,
      step.reward,
      if (step.endUp) 1.0 else 0.0,
      prevPos.row,
      prevPos.col,
      pos.row,
      pos.col)))
    val prevAgent = step.prevAgent.asInstanceOf[QAgent]
    val agent = step.agent.asInstanceOf[QAgent]
    val prevQ = prevAgent.q(prevEnv.observation)
    val prevQ1 = prevAgent.q(env.observation)
    val q1 = agent.q(env.observation)
    Nd4j.hstack(head, prevQ, q1, prevQ1)
  }

  def main(args: Array[String]) {
    val conf = loadConfig(if (args.isEmpty) "maze.yaml" else args(0))
    val numEpisodes = conf.getConf("session").getInt("numEpisodes").get
    val sync = conf.getConf("session").getLong("sync").get
    val mode = conf.getConf("session").getString("mode").get
    val dump = conf.getConf("session").getString("dump")
    val trace = conf.getConf("session").getString("trace")
    val model = conf.getConf("agent").getString("model")

    def onEpisode(episode: Episode) {
      val qagent = episode.agent.asInstanceOf[QAgent]
      model.foreach(qagent.writeModel)
      for {
        file <- dump
      } {
        val data = createDump(episode)
        withFile(file, true)(writeINDArray(_)(data))
      }
    }

    val session = Session(
      numEpisodes,
      buildEnv(conf),
      buildAgent(conf),
      sync = sync,
      mode = mode)
    session.episodeObs.subscribe(
      onEpisode(_),
      ex => logger.error(ex.getMessage, ex))
    trace.foreach(file => {
      session.stepObs.subscribe(step => {
        val data = createTrace(step)
        withFile(file, true)(writeINDArray(_)(data))
      })
    });
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
