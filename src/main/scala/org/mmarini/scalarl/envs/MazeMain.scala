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
import io.circe.Json

object MazeMain extends LazyLogging {
  private val ClearScreen = "\033[2J\033[H"

  private def buildEnv(conf: Json): Env = {
    val map = conf.hcursor.downField("env").get[Seq[String]]("map").right.get
    SimpleMazeEnv.fromStrings(map)
  }

  private def loadAgentConf(builder: AgentBuilder, agentCursor: ACursor) = {
    val builder1 = builder.
      numInputs(agentCursor.get[Int]("numInputs").right.get).
      numActions(agentCursor.get[Int]("numActions").right.get)

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

    val maxHistory = agentCursor.get[Int]("maxHistory").toOption
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

    builder17
  }

  private def buildAgent(conf: Json): Agent = {
    val agentCursor = conf.hcursor.downField("agent")

    val baseBuilder = AgentBuilder().
      agentType(agentCursor.
        get[String]("type").
        getOrElse("QAgent"))

    val builder = agentCursor.
      get[String]("loadModel").
      toOption.
      map(baseBuilder.file).
      getOrElse(loadAgentConf(baseBuilder, agentCursor))
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
    val file = if (args.isEmpty) "maze.yaml" else args(0)
    logger.info("File {}", file)
    val jsonConf = Configuration.jsonFromFile(file)
    val numSteps = jsonConf.hcursor.downField("session").get[Int]("numSteps").right.get
    val sync = jsonConf.hcursor.downField("session").get[Long]("sync").right.get
    val mode = jsonConf.hcursor.downField("session").get[String]("mode").right.get
    val dump = jsonConf.hcursor.downField("session").get[String]("dump").toOption
    val trace = jsonConf.hcursor.downField("session").get[String]("trace").toOption
    val saveModel = jsonConf.hcursor.downField("agent").get[String]("saveModel").toOption
    val maxEpisodeLength = jsonConf.hcursor.downField("session").get[Long]("maxEpisodeLength").getOrElse(Long.MaxValue)
    val sentinel = jsonConf.hcursor.downField("session").get[Boolean]("sentinel").getOrElse(false)
    Sentinel.activate(sentinel)

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
