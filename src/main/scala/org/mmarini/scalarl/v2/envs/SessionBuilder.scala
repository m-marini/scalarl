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

package org.mmarini.scalarl.v2.envs

import java.io.File

import com.typesafe.scalalogging.LazyLogging
import io.circe.ACursor
import monix.eval.Task
import monix.execution.Scheduler
import org.mmarini.scalarl.FileUtils.{withFile, writeINDArray}
import org.mmarini.scalarl.v2._
import org.mmarini.scalarl.v2.agents.ExpSarsaAgent
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.api.rng.Random
import org.nd4j.linalg.factory.Nd4j

/**
 *
 */
object SessionBuilder extends LazyLogging {

  /**
   * Returns the session
   *
   * @param conf  the json configuration
   * @param epoch the number of epoch
   * @param env   the environment
   * @param agent the agent
   */
  def fromJson(conf: ACursor)(epoch: Int, env: => Env, agent: => Agent): (Session, Random) = {
    val numSteps = conf.get[Int]("numSteps").right.get
    val dump = conf.get[String]("dump").toOption
    val trace = conf.get[String]("trace").toOption
    val saveModel = conf.get[String]("modelFile").toOption
    val random = conf.get[Long]("seed").map(
      Nd4j.getRandomFactory.getNewRandomInstance
    ).getOrElse(
      Nd4j.getRandom
    )

    // Clean up all files
    if (epoch == 0) {
      (dump.toSeq ++ trace ++ saveModel).foreach(new File(_).delete())
    }

    // Create session
    val session = new Session(
      numSteps = numSteps,
      env = env,
      agent = agent,
      epoch = epoch)

    // Create dump function
    val createDump: Step => INDArray = createLanderDump

    // Create tracing function
    val createTrace: Step => INDArray = createLanderTrace

    // Subscribe on step observable
    session.steps.doOnNext(step => Task.eval {
      onStep(trace, createTrace)(step)
      onStep(dump, createDump)(step)
    }).doOnError(ex => Task.eval {
      logger.error(ex.getMessage, ex)
    }).subscribe()(Scheduler.global)

    (session, random)
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
  private def createLanderDump(episode: Step): INDArray = {
    val kpi = Nd4j.create(Array[Double](episode.epoch, episode.step)).transpose()
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
    val env0 = step.env0.asInstanceOf[Lander]
    val pos0 = env0.pos
    val speed0 = env0.speed
    val env1 = step.env1.asInstanceOf[Lander]
    val pos1 = env1.pos
    val speed1 = env1.speed
    val head = Nd4j.create(Array[Double](
      step.epoch,
      step.step))
    val mid = Nd4j.create(Array(step.feedback.reward))
    val agent0 = step.agent0.asInstanceOf[ExpSarsaAgent]
    val agent1 = step.agent1.asInstanceOf[ExpSarsaAgent]
    val q0 = agent0.q(env0.observation)
    val q1 = agent0.q(env1.observation)
    val q01 = agent1.q(env0.observation)
    Nd4j.hstack(
      head,
      Nd4j.create(Array(step.feedback.action.toDouble)),
      mid,
      pos0,
      speed0,
      pos1,
      speed1,
      Nd4j.create(Array(agent0.avgReward)),
      q0,
      q1,
      q01)
  }

  /**
   *
   */
  def onStep(trace: Option[String], createTrace: Step => INDArray)(step: Step) {
    for {
      file <- trace
    } {
      val data = createTrace(step)
      withFile(file, append = true)(writeINDArray(data))
    }
  }
}
