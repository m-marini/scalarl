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

import scala.collection.JavaConversions.`deprecated seqAsJavaList`

import org.mmarini.scalarl.ActionChannelConfig
import org.mmarini.scalarl.Agent
import org.mmarini.scalarl.Env
import org.mmarini.scalarl.Episode
import org.mmarini.scalarl.FileUtils.withFile
import org.mmarini.scalarl.FileUtils.writeINDArray
import org.mmarini.scalarl.Session
import org.mmarini.scalarl.Step
import org.mmarini.scalarl.agents.AgentBuilder
import org.mmarini.scalarl.agents.PolicyFunction
import org.mmarini.scalarl.agents.TDAgent
import org.mmarini.scalarl.nn.Sentinel
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j

import com.typesafe.scalalogging.LazyLogging

import io.circe.ACursor
import org.mmarini.scalarl.agents.AgentNetworkBuilder

object Main extends LazyLogging {
  private val ClearScreen = "\033[2J\033[H"

  /**
   *
   */
  private def buildLanderConf(conf: ACursor): LanderConf = {
    val landerConf = LanderConf()
    val l1 = conf.get[Double]("h0Range").toOption.map(value =>
      landerConf.copy(h0Range = value)).getOrElse(landerConf)
    val l2 = conf.get[Double]("z0").toOption.map(value =>
      l1.copy(z0 = value)).getOrElse(l1)
    val l3 = conf.get[Double]("landingRadius").toOption.map(value =>
      l2.copy(landingRadius = value)).getOrElse(l2)
    val l4 = conf.get[Double]("landingVH").toOption.map(value =>
      l3.copy(landingVH = value)).getOrElse(l3)
    val l5 = conf.get[Double]("landingVZ").toOption.map(value =>
      l4.copy(landingVZ = value)).getOrElse(l4)
    val l6 = conf.get[Double]("dt").toOption.map(value =>
      l5.copy(dt = value)).getOrElse(l5)
    val l7 = conf.get[Double]("g").toOption.map(value =>
      l6.copy(g = value)).getOrElse(l6)
    val l8 = conf.get[Double]("maxAH").toOption.map(value =>
      l7.copy(maxAH = value)).getOrElse(l7)
    val l9 = conf.get[Double]("maxAZ").toOption.map(value =>
      l8.copy(maxAZ = value)).getOrElse(l8)
    val l10 = conf.get[Double]("vhSignalScale").toOption.map(value =>
      l9.copy(vhSignalScale = value)).getOrElse(l9)
    val l11 = conf.get[Double]("vzSignalScale").toOption.map(value =>
      l10.copy(vzSignalScale = value)).getOrElse(l10)
    val l12 = conf.get[Double]("landedReward").toOption.map(value =>
      l11.copy(landedReward = value)).getOrElse(l11)
    val l13 = conf.get[Double]("crashReward").toOption.map(value =>
      l12.copy(crashReward = value)).getOrElse(l12)
    val l14 = conf.get[Double]("outOfRangeReward").toOption.map(value =>
      l13.copy(outOfRangeReward = value)).getOrElse(l13)
    val l15 = conf.get[Double]("rewardDistanceScale").toOption.map(value =>
      l14.copy(rewardDistanceScale = value)).getOrElse(l14)
    val l16 = conf.get[Double]("outOfFuelReward").toOption.map(value =>
      l15.copy(outOfFuelReward = value)).getOrElse(l15)
    val l17 = conf.get[Int]("fuel").toOption.map(value =>
      l16.copy(fuel = value)).getOrElse(l16)
    l17
  }

  /**
   * 
   */
  private def buildEnv(conf: ACursor): Env =
    conf.get[String]("type").toOption match {
      case Some("Maze") =>
        val map = conf.get[Seq[String]]("map").right.get
        MazeEnv.fromStrings(map)
      case Some("Lander") =>
        val landerConf = buildLanderConf(conf)
        val seed = conf.get[Long]("seed").toOption
        val random = seed.map(seed =>
          Nd4j.getRandomFactory().getNewRandomInstance(seed)).
          getOrElse(Nd4j.getRandom())
        LanderStatus(
          conf = landerConf,
          random = random)
      case Some(typ) => throw new IllegalArgumentException(s"Unreconginzed env type '${typ}'")
      case _         => throw new IllegalArgumentException("Missing env type")
    }

  /**
   * 
   */
  private def buildAgent(
    agentCursor: ACursor,
    netCursor:   ACursor,
    config:      ActionChannelConfig): Agent = {
    val netBuilder = AgentNetworkBuilder(netCursor)
    AgentBuilder(agentCursor).networkBuilder(netBuilder).config(config).build()
  }

  /**
   * Returns the dump data array of the episode
   * The data array is composed by:
   *
   * - stepCount
   * - returnValue
   * - average loss
   * - 10 x 10 x 8 of q action values for each state for each action
   */
  private def createMazeDump(episode: Episode): INDArray = {
    val session = episode.session
    val agent = episode.agent.asInstanceOf[TDAgent]
    val kpi = Nd4j.create(Array(Array(episode.stepCount, episode.returnValue, episode.avgLoss)))
    val states = episode.env.asInstanceOf[MazeEnv].dumpStates
    val policy = states.map(agent.asInstanceOf[PolicyFunction].policy)
    val policyMat = Nd4j.vstack(policy).ravel()
    val mask = states.map(_.actions)
    val maskMat = Nd4j.vstack(mask).ravel()
    Nd4j.hstack(kpi, policyMat, maskMat)
  }

  /**
   * Returns the dump data array of sample
   * The data array is empty:
   */
  private def createLanderSample(step: Step): INDArray = {
    val env = step.beforeEnv.asInstanceOf[LanderStatus]
    val obs = env.observation
    val in = obs.signals
    val agent = step.beforeAgent.asInstanceOf[PolicyFunction]
    val q = agent.policy(obs)
    val action = step.action
    val reward = step.reward
    val endUp = if (step.endUp) 1.0 else 0.0
    Nd4j.hstack(
      in,
      q,
      action,
      Nd4j.create(Array(reward)),
      Nd4j.create(Array(endUp)))
  }

  /**
   * Returns the dump data array of sample
   * The data array is empty:
   */
  private def createMazeSample(step: Step): INDArray = {
    Nd4j.zeros(1)
  }

  /**
   * Returns the dump data array of the episode
   * The data array is composed by:
   *
   * - stepCount
   * - returnValue
   * - average loss
   * - 10 x 10 x 8 of q action values for each state for each action
   */
  private def createLanderDump(episode: Episode): INDArray = {
    val session = episode.session
    val agent = episode.agent.asInstanceOf[TDAgent]
    val kpi = Nd4j.create(Array(Array(episode.stepCount, episode.returnValue, episode.avgLoss)))
    Nd4j.hstack(kpi)
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
  private def createMazeTrace(step: Step): INDArray = {
    val beforeEnv = step.beforeEnv.asInstanceOf[MazeEnv]
    val beforePos = beforeEnv.subject
    val afterEnv = step.afterEnv.asInstanceOf[MazeEnv]
    val afterPos = afterEnv.subject
    val head = Nd4j.create(Array(Array[Double](
      step.episode,
      step.step)))
    val mid = Nd4j.create(Array(Array(
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
    Nd4j.hstack(head, step.action, mid, beforeQ, fitQ, afterQ, availableActions, afterAvailableActions)
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
  private def createLanderTrace(step: Step): INDArray = {
    val beforeEnv = step.beforeEnv.asInstanceOf[LanderStatus]
    val beforePos = beforeEnv.pos
    val beforeSpeed = beforeEnv.speed
    val afterEnv = step.afterEnv.asInstanceOf[LanderStatus]
    val afterPos = afterEnv.pos
    val afterSpeed = afterEnv.speed
    val head = Nd4j.create(Array(Array[Double](
      step.episode,
      step.step)))
    val mid = Nd4j.create(Array(Array(
      step.reward,
      if (step.endUp) 1 else 0)))
    val beforeAgent = step.beforeAgent.asInstanceOf[PolicyFunction]
    val afterAgent = step.afterAgent.asInstanceOf[PolicyFunction]
    val beforeQ = beforeAgent.policy(beforeEnv.observation)
    val afterQ = beforeAgent.policy(afterEnv.observation)
    val fitQ = afterAgent.policy(beforeEnv.observation)
    val availableActions = beforeEnv.observation.actions.ravel()
    val afterAvailableActions = afterEnv.observation.actions.ravel()
    Nd4j.hstack(
      head,
      step.action,
      mid,
      beforePos,
      beforeSpeed,
      beforeQ,
      afterPos,
      afterSpeed,
      fitQ,
      afterQ,
      availableActions,
      afterAvailableActions)
  }

  /**
   * 
   */
  def onEpisode(saveModel: Option[String], dump: Option[String], createDump: Episode => INDArray)(episode: Episode) {
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

  /**
   * 
   */
  def onStep(
    trace:        Option[String],
    createTrace:  Step => INDArray,
    samplesFile:  Option[String],
    createSample: Step => INDArray)(step: Step) {
    for {
      file <- trace
    } {
      val data = createTrace(step)
      withFile(file, true)(writeINDArray(_)(data))
    }
    for {
      file <- samplesFile
    } {
      val data = createSample(step)
      withFile(file, true)(writeINDArray(_)(data))
    }
  }

  /**
   * 
   */
  private def buildSession(
    sessionCursor: ACursor,
    env:           Env,
    agent:         Agent) = {
    val numSteps = sessionCursor.get[Int]("numSteps").right.get
    val sync = sessionCursor.get[Long]("sync").right.get
    val mode = sessionCursor.get[String]("mode").right.get
    val dump = sessionCursor.get[String]("dump").toOption
    val trace = sessionCursor.get[String]("trace").toOption
    val samplesFile = sessionCursor.get[String]("samples").toOption
    val saveModel = sessionCursor.get[String]("modelFile").toOption
    val maxEpisodeLength = sessionCursor.get[Long]("maxEpisodeLength").getOrElse(Long.MaxValue)
    val sentinel = sessionCursor.get[Boolean]("sentinel").getOrElse(false)

    Sentinel.activate(sentinel)

    // Clean up all files
    (dump.toSeq ++ trace ++ saveModel).foreach(new File(_).delete())

    // Create session
    val session = Session(
      noSteps = numSteps,
      env0 = env,
      agent0 = agent,
      maxEpisodeLength = maxEpisodeLength)

    // Create dump function
    val createDump: Episode => INDArray = if (env.isInstanceOf[Maze]) {
      createMazeDump _
    } else {
      createLanderDump _
    }

    // Subscribe on episode observable
    session.episodeObs.subscribe(
      onEpisode(saveModel, dump, createDump),
      ex => logger.error(ex.getMessage, ex))

    // Create tracing function
    val createTrace: Step => INDArray = if (env.isInstanceOf[Maze]) {
      createMazeTrace _
    } else {
      createLanderTrace _
    }

    // Create sampling function
    val createSamples: Step => INDArray = if (env.isInstanceOf[Maze]) {
      createMazeSample _
    } else {
      createLanderSample _
    }

    // Subscribe on step observable
    session.stepObs.subscribe(
      onStep(trace, createTrace, samplesFile, createSamples),
      ex => logger.error(ex.getMessage, ex))

    session
  }

  /**
   * 
   */
  def main(args: Array[String]) {
    val file = if (args.isEmpty) "maze.yaml" else args(0)
    logger.info("File {}", file)

    val jsonConf = Configuration.jsonFromFile(file)

    val env = buildEnv(jsonConf.hcursor.downField("env"))
    val agent = buildAgent(
      jsonConf.hcursor.downField("agent"),
      jsonConf.hcursor.downField("network"),
      env.actionConfig)
    val session = buildSession(
      jsonConf.hcursor.downField("session"),
      env = env,
      agent = agent)

    val (env1, agent1) = session.run()
    logger.info("Session completed.")
  }
}
