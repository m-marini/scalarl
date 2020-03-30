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

package org.mmarini.scalarl.ts.envs

import java.io.File

import com.typesafe.scalalogging.LazyLogging
import io.circe.ACursor
import monix.eval.Task
import monix.execution.Scheduler
import org.mmarini.scalarl.FileUtils.{withFile, writeINDArray}
import org.mmarini.scalarl.ts._
import org.mmarini.scalarl.ts.agents.DynaQPlusAgent
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.api.rng.Random
import org.nd4j.linalg.factory.Nd4j

class SessionBuilder(sessionCursor: ACursor) extends LazyLogging {

  /**
   *
   */
  def build(env: Env, agent: Agent): (Session, Random) = {
    val numSteps = sessionCursor.get[Int]("numSteps").right.get
    val dump = sessionCursor.get[String]("dump").toOption
    val trace = sessionCursor.get[String]("trace").toOption
    val samplesFile = sessionCursor.get[String]("samples").toOption
    val saveModel = sessionCursor.get[String]("modelFile").toOption
    val maxEpisodeLength = sessionCursor.get[Long]("maxEpisodeLength").getOrElse(Long.MaxValue)
    val random = sessionCursor.get[Long]("seed").map(
      Nd4j.getRandomFactory.getNewRandomInstance
    ).getOrElse(
      Nd4j.getRandom
    )

    // Clean up all files
    (dump.toSeq ++ trace ++ saveModel).foreach(new File(_).delete())

    // Create session
    val session = Session(
      noSteps = numSteps,
      env0 = env,
      agent0 = agent,
      maxEpisodeLength = maxEpisodeLength)

    // Create dump function
    val createDump: Episode => INDArray = createLanderDump

    // Subscribe on episode observable
    session.episodes.doOnNext(episode => Task.eval {
      onEpisode(saveModel, dump, createDump)(episode)
    }).doOnError(ex => Task.eval {
      logger.error(ex.getMessage, ex)
    }).subscribe()(Scheduler.global)

    // Create tracing function
    val createTrace: Step => INDArray = createLanderTrace

    // Create sampling function
    val createSamples: Step => INDArray = createLanderSample

    // Subscribe on step observable
    session.steps.doOnNext(step => Task.eval {
      onStep(trace, createTrace, samplesFile, createSamples)(step)
    }).doOnError(ex => Task.eval {
      logger.error(ex.getMessage, ex)
    }).subscribe()(Scheduler.global)

    (session, random)
  }

  /**
   * Returns the dump data array of sample
   * The data array is empty:
   */
  private def createLanderSample(step: Step): INDArray = {
    val env = step.beforeEnv.asInstanceOf[LanderStatus]
    val obs = env.observation
    val in = obs.signals
    val action = step.feedback.action
    val reward = step.feedback.reward
    val endUp = if (step.feedback.s1.endUp) 1.0 else 0.0
    Nd4j.hstack(
      in,
      action,
      Nd4j.create(Array(reward)),
      Nd4j.create(Array(endUp)))
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
      step.feedback.reward,
      if (step.feedback.s1.endUp) 1 else 0)))
    val beforeAgent = step.beforeAgent.asInstanceOf[DynaQPlusAgent]
    val afterAgent = step.afterAgent.asInstanceOf[DynaQPlusAgent]
    val beforeQ = beforeAgent.policy(beforeEnv.observation)
    val afterQ = beforeAgent.policy(afterEnv.observation)
    val fitQ = afterAgent.policy(beforeEnv.observation)
    val availableActions = beforeEnv.observation.actions.ravel()
    val afterAvailableActions = afterEnv.observation.actions.ravel()
    Nd4j.hstack(
      head,
      step.feedback.action,
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
  def onEpisode(saveModel: Option[String],
                dump: Option[String],
                createDump: Episode => INDArray)(episode: Episode) {
    for {
      file <- saveModel
    } {
      episode.agent.writeModel(file)
    }
    for {
      file <- dump
    } {
      val data = createDump(episode)
      withFile(file, append = true)(writeINDArray(data))
    }
    logger.info(f"SessionStep ${
      episode.step
    }%,6d Episode ${
      episode.episode
    }%,6d, Steps ${
      episode.stepCount
    }%,6d, loss=${
      episode.avgLoss
    }%12g ,returns=${
      episode.returnValue
    }%12g")
  }

  /**
   *
   */
  def onStep(trace: Option[String],
             createTrace: Step => INDArray,
             samplesFile: Option[String],
             createSample: Step => INDArray)(step: Step) {
    for {
      file <- trace
    } {
      val data = createTrace(step)
      withFile(file, append = true)(writeINDArray(data))
    }
    for {
      file <- samplesFile
    } {
      val data = createSample(step)
      withFile(file, append = true)(writeINDArray(data))
    }
  }
}

/**
 * Factory of [[SessionBuilder]]
 */
object SessionBuilder {

  /**
   * Returns the session builder
   *
   * @param conf the configuration
   */
  def apply(conf: ACursor): SessionBuilder = new SessionBuilder(conf)
}