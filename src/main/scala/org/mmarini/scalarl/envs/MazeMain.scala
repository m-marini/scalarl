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
import org.mmarini.scalarl.agents.AgentBuilder
import org.mmarini.scalarl.agents.AgentType
import org.mmarini.scalarl.agents.PolicyFunction
import org.mmarini.scalarl.agents.TDAgent
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j

import com.typesafe.scalalogging.LazyLogging

import io.circe.Json

object MazeMain extends LazyLogging {
  private val ClearScreen = "\033[2J\033[H"

  private def buildEnv(conf: Json): Env = {
    val map = conf.hcursor.downField("env").get[Seq[String]]("map").right.get
    SimpleMazeEnv.fromStrings(map)
  }

  private def buildAgent(conf: Json): Agent = {
    val numInputs = conf.hcursor.downField("agent").get[Int]("numInputs").right.get
    val numActions = conf.hcursor.downField("agent").get[Int]("numActions").right.get
    val numHiddens = conf.hcursor.downField("agent").get[List[Int]]("numHiddens").right.get
    val seed = conf.hcursor.downField("agent").get[Long]("seed").getOrElse(0L)
    val epsilon = conf.hcursor.downField("agent").get[Double]("epsilon").right.get
    val gamma = conf.hcursor.downField("agent").get[Double]("gamma").right.get
    val lambda = conf.hcursor.downField("agent").get[Double]("lambda").getOrElse(0.0)
    val kappa = conf.hcursor.downField("agent").get[Double]("kappa").getOrElse(1.0)
    val learningRate = conf.hcursor.downField("agent").get[Double]("learningRate").right.get
    val maxAbsGrads = conf.hcursor.downField("agent").get[Double]("maxAbsGradients").right.get
    val maxAbsParams = conf.hcursor.downField("agent").get[Double]("maxAbsParameters").right.get
    val loadModel = conf.hcursor.downField("agent").get[String]("loadModel").toOption
    //    val traceUpdater = conf.hcursor.downField("agent").getString("traceUpdater").map(TraceUpdater.fromString).getOrElse(AccumulateTraceUpdater)
    val agentType = conf.hcursor.downField("agent").get[String]("type").map(AgentType.withName).getOrElse(AgentType.QAgent)

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
      agentType(agentType)
    //      traceUpdater(traceUpdater)
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
    val states = episode.env.asInstanceOf[SimpleMazeEnv].dumpStates
    val policy = states.map(agent.asInstanceOf[PolicyFunction].policy)
    val policyMat = Nd4j.vstack(policy).ravel()
    val mask = states.map(_.actions)
    val maskMat = Nd4j.vstack(mask).ravel()
    Nd4j.hstack(kpi, policyMat, maskMat)
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
    val beforeEnv = step.beforeEnv.asInstanceOf[SimpleMazeEnv]
    val beforePos = beforeEnv.subject
    val afterEnv = step.afterEnv.asInstanceOf[SimpleMazeEnv]
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
    val afterAvailableActions = afterEnv.observation.actions.ravel()
    Nd4j.hstack(head, beforeQ, fitQ, afterQ, availableActions, afterAvailableActions)
  }

  def onEpisode(saveModel: Option[String], dump: Option[String])(episode: Episode) {
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
    logger.info(f"SessionStep ${
      episode.step
    }%,d Episode ${
      episode.episode
    }%,d, Steps ${
      episode.stepCount
    }%,d, loss=${
      episode.avgLoss
    }%g ,returns=${
      episode.returnValue
    }%g")
  }

  def onStep(trace: Option[String])(step: Step) {
    for {
      file <- trace
    } {
      val data = createTrace(step)
      withFile(file, true)(writeINDArray(_)(data))
    }
  }

  def main(args: Array[String]) {
    val jsonConf = Configuration.jsonFromFile(if (args.isEmpty) "maze.yaml" else args(0))
    val numSteps = jsonConf.hcursor.downField("session").get[Int]("numSteps").right.get
    val sync = jsonConf.hcursor.downField("session").get[Long]("sync").right.get
    val mode = jsonConf.hcursor.downField("session").get[String]("mode").right.get
    val dump = jsonConf.hcursor.downField("session").get[String]("dump").toOption
    val trace = jsonConf.hcursor.downField("session").get[String]("trace").toOption
    val saveModel = jsonConf.hcursor.downField("agent").get[String]("saveModel").toOption
    val maxEpisodeLength = jsonConf.hcursor.downField("session").get[Long]("maxEpisodeLength").getOrElse(Long.MaxValue)

    (dump.toSeq ++ trace).foreach(new File(_).delete())

    val session = Session(
      noSteps = numSteps,
      env0 = buildEnv(jsonConf),
      agent0 = buildAgent(jsonConf),
      maxEpisodeLength = maxEpisodeLength)

    session.episodeObs.subscribe(
      onEpisode(saveModel, dump),
      ex => logger.error(ex.getMessage, ex))

    session.stepObs.subscribe(
      onStep(trace),
      ex => logger.error(ex.getMessage, ex))

    session.run()
  }
}
