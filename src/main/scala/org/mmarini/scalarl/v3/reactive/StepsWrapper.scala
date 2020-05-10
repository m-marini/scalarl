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

package org.mmarini.scalarl.v3.reactive

import java.io.File

import monix.eval.Task
import monix.reactive.Observable
import org.mmarini.scalarl.v3.Utils._
import org.mmarini.scalarl.v3.agents.{ActorCriticAgent, PolicyActor, PriorityPlanner}
import org.mmarini.scalarl.v3.envs.LanderStatus
import org.mmarini.scalarl.v3.{Feedback, Step}
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j._
import org.nd4j.linalg.ops.transforms.Transforms._

/**
 * Wrapper on step observable
 *
 * @param observable the observable
 */
class StepsWrapper(val observable: Observable[Step]) extends ObservableWrapper[Step] {

  /** Returns the monitored obbservable */
  def monitorInfo(): MonitorWrapper = {
    val kpi = observable.mapAccumulate(zeros(2)) {
      case (seed, step) =>
        val newSeed = seed.add(hstack(step.feedback.reward, step.score))
        val count = step.step
        val kpis = newSeed.div(count)
        val result = (step, kpis)
        (newSeed, result)
    }
    new MonitorWrapper(kpi)
  }

  /**
   * Returns the kpis
   * The kpis consists of
   * - critic eta
   * - actors
   *  - alpha
   *  - eta
   */
  def kpis(): INDArrayWrapper = new INDArrayWrapper(observable.map(step => {
    val Feedback(s0, actions, reward, s1) = step.feedback
    val agent = step.agent0.asInstanceOf[ActorCriticAgent]
    val agent1 = step.agent1.asInstanceOf[ActorCriticAgent]
    val v0 = agent.v(s0)
    val v1 = agent.v(s1)
    val v01 = agent1.v(s0)
    val (delta, vStar, _) = agent.computeDelta(v0, v1, reward)
    val deltaCritic = vStar.distance2(v0)
    val deltaCritic1 = vStar.distance2(v01)
    val actorsKpis = (for {
      ((a0, a1), action) <- agent.actors.zip(agent1.actors).zipWithIndex.toSeq
    } yield {
      (a0, a1) match {
        case (actor: PolicyActor, actor1: PolicyActor) =>
          val pr = actor.preferences(s0)
          val prStar = PolicyActor.computeActorLabel(pr, actions.getInt(action), actor.alpha, delta)
          val pr1 = actor1.preferences(s0)
          val deltaActor = prStar.distance2(pr)
          val deltaActor1 = prStar.distance2(pr1)
          Seq(deltaActor * deltaActor, deltaActor1 * deltaActor1)
        case _ => Seq()
      }
    }).flatten
    val plannerKpis = agent.planner match {
      case Some(pl: PriorityPlanner[INDArray, INDArray]) =>
        Seq(pl.model.data.size.toDouble, pl.queue.queue.size.toDouble)
      case _ => Seq()
    }
    val kpis = create((
      step.epoch.toDouble +:
        step.step.toDouble +:
        deltaCritic * deltaCritic +:
        deltaCritic1 * deltaCritic1 +:
        (actorsKpis ++
          plannerKpis)).toArray)

    kpis
  }))

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
    val LanderStatus(_, pos0, speed0, t0, fuel0, _, _) = env0
    val Feedback(o0, actions, reward, _) = feedback
    val s0 = env0.asInstanceOf[LanderStatus]
    val signalsId = find(o0.signals).hashCode() & ((1 << 12) - 1)
    val agent = step.agent0.asInstanceOf[ActorCriticAgent]

    val prefs = hstack(agent.actors.map {
      case actor: PolicyActor =>
        val pr = actor.preferences(o0)
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
      prefs,
      softmax(prefs))
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
