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

import com.typesafe.scalalogging.LazyLogging
import io.circe.ACursor
import java.io.File
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
import scala.collection.JavaConversions.`deprecated seqAsJavaList`
import com.sun.jndi.ldap.Ber

object Main extends LazyLogging {
  private val ClearScreen = "\033[2J\033[H"

  private def buildEnv(conf: ACursor): Env =
    conf.get[String]("type").toOption match {
      case Some("Maze") =>
        val map = conf.get[Seq[String]]("map").right.get
        MazeEnv.fromStrings(map)
      case Some("Lander") =>
        val hRange = conf.get[Double]("hRange").right.get
        val height = conf.get[Double]("height").right.get
        val seed = conf.get[Long]("seed").toOption
        val random = seed.map(seed =>
          Nd4j.getRandomFactory().getNewRandomInstance(seed)).
          getOrElse(Nd4j.getRandom())
        LanderStatus.apply(
          hRange = hRange,
          height = height,
          random = random)
      case Some(typ) => throw new IllegalArgumentException(s"Unreconginzed env type '${typ}'")
      case _         => throw new IllegalArgumentException("Missing env type")
    }

  private def loadAgentConf(
    builder:     AgentBuilder,
    agentCursor: ACursor,
    config:      ActionChannelConfig) = {
    val builder1 = builder.
      numInputs(agentCursor.get[Int]("numInputs").right.get).
      config(config)

    val seed = agentCursor.get[Long]("seed").toOption
    val numHiddens = agentCursor.get[List[Int]]("numHiddens").toOption
    val epsilon = agentCursor.get[Double]("epsilon").toOption
    val gamma = agentCursor.get[Double]("gamma").toOption
    val kappa = agentCursor.get[Double]("kappa").toOption

    val trace = agentCursor.get[String]("trace").toOption
    val lambda = agentCursor.get[Double]("lambda").toOption

    val optimizer = agentCursor.get[String]("optimizer").toOption
    val learningRate = agentCursor.get[Double]("learningRate").toOption
    val beta1 = agentCursor.get[Double]("beta1").toOption
    val beta2 = agentCursor.get[Double]("beta2").toOption
    val epsilonAdam = agentCursor.get[Double]("epsilonAdam").toOption

    val maxAbsGrads = agentCursor.get[Double]("maxAbsGradients").toOption
    val maxAbsParams = agentCursor.get[Double]("maxAbsParameters").toOption

    val minHistory = agentCursor.get[Int]("minHistory").toOption
    val maxHistory = agentCursor.get[Int]("maxHistory").toOption
    val stepInterval = agentCursor.get[Int]("stepInterval").toOption
    val numBootstrapIteration = agentCursor.get[Int]("numBootstrapIteration").toOption
    val numBatchIteration = agentCursor.get[Int]("numBatchIteration").toOption

    val builder2 = seed.map(builder1.seed).getOrElse(builder1)
    val builder3 = numHiddens.map(builder2.numHiddens).getOrElse(builder2)
    val builder4 = epsilon.map(builder3.epsilon).getOrElse(builder3)
    val builder5 = gamma.map(builder4.gamma).getOrElse(builder4)
    val builder6 = kappa.map(builder5.kappa).getOrElse(builder5)

    val builder7 = trace.map(builder6.trace).getOrElse(builder6)
    val builder8 = lambda.map(builder7.lambda).getOrElse(builder7)

    val builder9 = optimizer.map(builder8.optimizer).getOrElse(builder8)
    val builder10 = learningRate.map(builder9.learningRate).getOrElse(builder9)
    val builder11 = beta1.map(builder10.beta1).getOrElse(builder10)
    val builder12 = beta2.map(builder11.beta2).getOrElse(builder11)
    val builder13 = epsilonAdam.map(builder12.epsilonAdam).getOrElse(builder12)

    val builder14 = maxAbsGrads.map(builder13.maxAbsGradient).getOrElse(builder13)
    val builder15 = maxAbsParams.map(builder14.maxAbsParams).getOrElse(builder14)

    val builder16 = maxHistory.map(builder15.maxHistory).getOrElse(builder15)
    val builder17 = numBatchIteration.map(builder16.numBatchIteration).getOrElse(builder16)
    val builder18 = minHistory.map(builder17.minHistory).getOrElse(builder17)
    val builder19 = stepInterval.map(builder18.stepInterval).getOrElse(builder18)
    val builder20 = numBootstrapIteration.map(builder19.numBootstrapIteration).getOrElse(builder19)

    builder20
  }

  private def buildAgent(agentCursor: ACursor, config: ActionChannelConfig): Agent = {
    val baseBuilder = AgentBuilder().
      agentType(agentCursor.
        get[String]("type").
        getOrElse("QAgent"))

    val builder = agentCursor.
      get[String]("loadModel").
      toOption.
      map(baseBuilder.file).
      getOrElse(loadAgentConf(baseBuilder, agentCursor, config))
    builder.build()
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

  def onStep(trace: Option[String], createTrace: Step => INDArray)(step: Step) {
    for {
      file <- trace
    } {
      val data = createTrace(step)
      withFile(file, true)(writeINDArray(_)(data))
    }
  }

  private def buildSession(
    sessionCursor: ACursor,
    env:           Env,
    agent:         Agent) = {
    val numSteps = sessionCursor.get[Int]("numSteps").right.get
    val sync = sessionCursor.get[Long]("sync").right.get
    val mode = sessionCursor.get[String]("mode").right.get
    val dump = sessionCursor.get[String]("dump").toOption
    val trace = sessionCursor.get[String]("trace").toOption
    //    val saveModel = jsonConf.hcursor.downField("agent").get[String]("saveModel").toOption
    val maxEpisodeLength = sessionCursor.get[Long]("maxEpisodeLength").getOrElse(Long.MaxValue)
    val sentinel = sessionCursor.get[Boolean]("sentinel").getOrElse(false)
    Sentinel.activate(sentinel)

    (dump.toSeq ++ trace).foreach(new File(_).delete())

    val session = Session(
      noSteps = numSteps,
      env0 = env,
      agent0 = agent,
      maxEpisodeLength = maxEpisodeLength)

    val createDump: Episode => INDArray = if (env.isInstanceOf[Maze])
      createMazeDump _
    else
      createLanderDump _

    session.episodeObs.subscribe(
      onEpisode(None, dump, createDump),
      ex => logger.error(ex.getMessage, ex))

    val createTrace: Step => INDArray = if (env.isInstanceOf[Maze])
      createMazeTrace _
    else
      createLanderTrace _

    session.stepObs.subscribe(
      onStep(trace, createTrace),
      ex => logger.error(ex.getMessage, ex))
    session
  }

  def main(args: Array[String]) {
    val file = if (args.isEmpty) "maze.yaml" else args(0)
    logger.info("File {}", file)

    val jsonConf = Configuration.jsonFromFile(file)

    val env = buildEnv(jsonConf.hcursor.downField("env"))
    val agent = buildAgent(
      jsonConf.hcursor.downField("agent"),
      env.actionConfig)
    val session = buildSession(
      jsonConf.hcursor.downField("session"),
      env = env,
      agent = agent)

    session.run()
  }
}
