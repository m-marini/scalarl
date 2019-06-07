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

import java.io.File

import scala.collection.JavaConversions.`deprecated mapAsScalaMap`
import scala.collection.JavaConversions.`deprecated seqAsJavaList`

import org.mmarini.scalarl.Agent
import org.mmarini.scalarl.Env
import org.mmarini.scalarl.Episode
import org.mmarini.scalarl.FileUtils.withFile
import org.mmarini.scalarl.FileUtils.writeINDArray
import org.mmarini.scalarl.Session
import org.mmarini.scalarl.Step
import org.mmarini.scalarl.agents.AccumulateTraceUpdater
import org.mmarini.scalarl.agents.AgentType
import org.mmarini.scalarl.agents.PolicyFunction
import org.mmarini.scalarl.agents.TDAgent
import org.mmarini.scalarl.agents.TraceUpdater
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j

import com.typesafe.scalalogging.LazyLogging
import org.mmarini.scalarl.agents.AgentBuilder

object MazeMain extends LazyLogging {
  private val ClearScreen = "\033[2J\033[H"

  private def buildEnv(conf: Configuration): Env = {
    val map = conf.getConf("env").getList[String]("map")
    MazeEnv.fromStrings(map)
  }

  private def buildAgent(conf: Configuration): Agent = {
    val numInputs = conf.getConf("agent").getInt("numInputs").get
    val numActions = conf.getConf("agent").getInt("numActions").get
    val numHiddens = conf.getConf("agent").getList[Int]("numHiddens")
    val seed = conf.getConf("agent").getLong("seed").getOrElse(0L)
    val epsilon = conf.getConf("agent").getDouble("epsilon").get
    val gamma = conf.getConf("agent").getDouble("gamma").get
    val lambda = conf.getConf("agent").getDouble("lambda").getOrElse(0.0)
    val kappa = conf.getConf("agent").getDouble("kappa").getOrElse(1.0)
    val learningRate = conf.getConf("agent").getDouble("learningRate").get
    val maxAbsGrads = conf.getConf("agent").getDouble("maxAbsGradients").get
    val maxAbsParams = conf.getConf("agent").getDouble("maxAbsParameters").get
    val loadModel = conf.getConf("agent").getString("loadModel")
    val traceUpdater = conf.getConf("agent").getString("traceUpdater").map(TraceUpdater.fromString).getOrElse(AccumulateTraceUpdater)
    val agentType = conf.getConf("agent").getString("type").map(AgentType.withName).getOrElse(AgentType.QAgent)

    val baseBuilder = AgentBuilder().
      numInputs(numInputs).
      numActions(numActions).
      numHiddens(numHiddens: _*).
      epsilon(epsilon).
      gamma(gamma).
      lambda(lambda).
      kappa(kappa).
      learningRate(learningRate).
      maxAbsGradient(maxAbsGrads).
      maxAbsParams(maxAbsParams).
      seed(seed).
      agentType(agentType).
      traceUpdater(traceUpdater)
    loadModel.
      map(baseBuilder.file).
      getOrElse(baseBuilder).
      build()
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
    val agent = episode.agent.asInstanceOf[TDAgent]
    val kpi = Nd4j.create(Array(Array(episode.stepCount, episode.returnValue, episode.avgLoss)))
    val states = episode.env.asInstanceOf[MazeEnv].dumpStates
    val policy = states.map(agent.asInstanceOf[PolicyFunction].policy)
    val policyMat = Nd4j.vstack(policy).ravel()
    Nd4j.hstack(kpi, policyMat)
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
    val beforeEnv = step.beforeEnv.asInstanceOf[MazeEnv]
    val beforePos = beforeEnv.subject
    val afterEnv = step.afterEnv.asInstanceOf[MazeEnv]
    val afterPos = afterEnv.subject
    val head = Nd4j.create(Array(Array(
      step.episode,
      step.step,
      step.action,
      step.reward,
      if (step.endUp) 1 else 0,
      beforePos.row,
      beforePos.col,
      afterPos.row,
      afterPos.col)))
    val beforeAgent = step.beforeAgent.asInstanceOf[PolicyFunction]
    val afterAgent = step.afterAgent.asInstanceOf[PolicyFunction]
    val beforeQ = beforeAgent.policy(beforeEnv.observation)
    val afterQ = beforeAgent.policy(afterEnv.observation)
    val fitQ = afterAgent.policy(beforeEnv.observation)
    val availableActions = beforeEnv.observation.actions.ravel()
    Nd4j.hstack(head, beforeQ, fitQ, afterQ, availableActions)
  }

  def main(args: Array[String]) {
    val conf = Configuration.fromFile(if (args.isEmpty) "maze.yaml" else args(0))
    val numEpisodes = conf.getConf("session").getInt("numEpisodes").get
    val sync = conf.getConf("session").getLong("sync").get
    val mode = conf.getConf("session").getString("mode").get
    val dump = conf.getConf("session").getString("dump")
    val trace = conf.getConf("session").getString("trace")
    val saveModel = conf.getConf("agent").getString("saveModel")
    val maxEpisodeLength = conf.getConf("session").getLong("maxEpisodeLength").getOrElse(Long.MaxValue)

    def onEpisode(episode: Episode) {
      for {
        file <- saveModel
      } {
        episode.agent.writeModel(file)
      }
      for {
        file <- dump
      } {
        val data = createDump(episode)
        withFile(file, true)(writeINDArray(_)(data))
      }
      println(s"Episode ${episode.episode}, Steps ${episode.stepCount}, loss=${episode.avgLoss} ")
    }

    def render(step: Step) {
      val Step(episodeCount, stepCount, _, _, endUp, _, _, env, _, _) = step
      mode match {
        case "human" =>
          print(ClearScreen + "\r")
          env.asInstanceOf[MazeEnv].render()
          print(s"\nEpisode ${episodeCount} / Step ${stepCount}")
          System.out.flush()
        case "stats" =>
          if (endUp) {
            println(s"Episode ${episodeCount} / Step ${stepCount}")
            System.out.flush()
          }
        case _ =>
          print(ClearScreen + s"\rEpisode ${episodeCount} / Step ${stepCount}")
          System.out.flush()
      }
    }

    def onStep(step: Step) {
      //      render(step)
      for {
        file <- trace
      } {
        val data = createTrace(step)
        withFile(file, true)(writeINDArray(_)(data))
      }
    }

    (dump.toSeq ++ trace).foreach(new File(_).delete())

    dump.foreach(new File(_).delete())
    val session = Session(
      noEpisode = numEpisodes,
      env0 = buildEnv(conf),
      agent0 = buildAgent(conf),
      maxEpisodeLength = maxEpisodeLength)

    session.episodeObs.subscribe(
      onEpisode(_),
      ex => logger.error(ex.getMessage, ex))

    session.stepObs.subscribe(
      onStep(_),
      ex => logger.error(ex.getMessage, ex))

    session.run()
  }
}
