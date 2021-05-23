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

package org.mmarini.scalarl.v6.reactive

import monix.eval.Task
import monix.reactive.Observable
import org.mmarini.scalarl.v6.Utils._
import org.mmarini.scalarl.v6.agents.{ActorCriticAgent, PolicyActor}
import org.mmarini.scalarl.v6.envs.LanderStatus
import org.mmarini.scalarl.v6.{Feedback, Step}
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j._
import org.nd4j.linalg.ops.transforms.Transforms._

import java.io.File

/**
 * Wrapper on step observable
 *
 * @param observable the observable
 */
class StepsWrapper(val observable: Observable[Step]) extends ObservableWrapper[Step] {

  /** Returns the monitored observable */
  def monitorInfo(): MonitorWrapper = {
    val kpi = observable.mapAccumulate(zeros(2)) {
      case (seed, step) =>
        val newSeed = seed.add(hstack(step.feedback.reward, step.score))
        val count = step.step + 1
        val kpis = newSeed.div(count)
        val result = (step, kpis)
        (newSeed, result)
    }
    new MonitorWrapper(kpi)
  }

  /**
   * Returns the trace observable
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
   *
   */
  def landerTrace(): INDArrayWrapper = new INDArrayWrapper(observable.filter(_.env0 match {
    case _: LanderStatus => true
    case _ => false
  }).map(createLanderTrace))

  /**
   * Returns the trace data array of the step
   *
   * @param step the step
   */
  private def createLanderTrace(step: Step): INDArray = {
    val Step(epoch, stepId, feedback, env0, _, _, _, score, _, _) = step
    val LanderStatus(pos0, speed0, t0, fuel0, _) = env0
    val Feedback(o0, actions, reward, _) = feedback
    val s0 = env0.asInstanceOf[LanderStatus]
    val signalsId = find(o0.signals).hashCode() & ((1 << 12) - 1)
    val agent = step.agent0.asInstanceOf[ActorCriticAgent]
    val out = agent.network.output(agent.conf.stateEncode(o0.signals))

    val preferences = hstack(agent.conf.actors.map {
      case actor: PolicyActor =>
        val pr = actor.preferences(out)
        val pi = softmax(pr)
        hstack(pr, pi)
      case _ =>
        throw new IllegalArgumentException("wrong actor type")
    }: _*)

    val trace = hstack(create(
      Array[Double](
        epoch,
        stepId,
        signalsId,
        s0.status.id)),
      pos0,
      speed0,
      fuel0,
      t0,
      actions,
      reward,
      score,
      preferences)
    trace
  }

  /**
   * Returns the lander dump
   * The data array is composed by:
   *
   * - epoch
   * - step
   * - returnValue
   * - score
   *
   */
  def landerDump(): INDArrayWrapper =
    new INDArrayWrapper(
      observable.filter(_.env0 match {
        case _: LanderStatus => true
        case _ => false
      }).map(createLanderDump))

  /**
   * Returns the dump data array of the episode
   *
   * @param step the step
   */
  private def createLanderDump(step: Step): INDArray = hstack(create(
    Array[Double](
      step.epoch,
      step.step)),
    step.feedback.reward,
    step.score)

  /**
   * Returns the steps observable that saves the model
   *
   * @param path the model path
   */
  def saveAgent(path: File): StepsWrapper = new StepsWrapper(
    observable.doOnNext(step => Task.eval {
      logger.info("saving model {}", path)
      step.context.agent.writeModel(path)
    })
  )

  /** Returns the lander observable */
  def lander(): LanderWrapper = new LanderWrapper(
    observable.filter(_.env0 match {
      case _: LanderStatus => true
      case _ => false
    }).map(_.env0.asInstanceOf[LanderStatus]))
}
