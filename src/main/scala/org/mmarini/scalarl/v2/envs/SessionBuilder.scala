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
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j

import scala.concurrent.duration.DurationInt

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
  def fromJson(conf: ACursor)(epoch: Int, env: => Env, agent: => Agent): Session = {
    val numSteps = conf.get[Int]("numSteps").toTry.get
    val dump = conf.get[String]("dump").toOption
    val trace = conf.get[String]("trace").toOption
    val saveModel = conf.get[String]("modelFile").toOption

    // Clean up all files
    if (epoch == 0) {
      (dump.toSeq ++ trace ++ saveModel).foreach(new File(_).delete())
    }
    saveModel.foreach(f => {
      new File(f).mkdirs()
    })


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

    // log data every 2 seconds
    session.steps.sample(2 seconds).doOnNext(step => Task.eval {
      logger.info(f"Epoch ${
        step.epoch
      }%,3d, Steps ${
        step.step
      }%6d ,avg rewards=${
        if (step.step != 0) step.context.returnValue / step.step else 0.0
      }%,12g, avgScore=${
        if (step.step != 0) step.context.totalScore / step.step else 0.0
      }%12g")
    }).subscribe()(Scheduler.global)

    // Save model every 10 steps
    session.steps.takeEveryNth(10).doOnNext(step => Task.eval {
      saveModel.foreach(step.context.agent.writeModel)
    }).doOnError(ex => Task.eval {
      logger.error(ex.getMessage, ex)
    }).subscribe()(Scheduler.global)

    // Save at the end
    session.steps.last.doOnNext(step => Task.eval {
      saveModel.foreach(step.context.agent.writeModel)
    }).doOnError(ex => Task.eval {
      logger.error(ex.getMessage, ex)
    }).subscribe()(Scheduler.global)

    session
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
  private def createLanderDump(step: Step): INDArray =
    Nd4j.create(Array[Double](
      step.epoch,
      step.step,
      step.feedback.reward,
      step.score))

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
    val Step(epoch, stepCount, feedback, env0, _, env1, _, score, _, _) = step
    val LanderStatus(_, pos0, speed0, t0, fuel0, _) = env0
    val LanderStatus(_, pos1, speed1, _, _, _) = env1
    val isFinal = if (env1.asInstanceOf[LanderStatus].isFinal) 1.0 else 0.0
    val Feedback(_, action, reward, _) = feedback

    val result = Nd4j.hstack(
      Nd4j.create(Array[Double](
        epoch,
        stepCount,
        reward,
        score,
        t0,
        fuel0)),
      pos0,
      speed0,
      Nd4j.create(Array[Double](
        action,
        isFinal)),
      pos1,
      speed1)
    result
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
