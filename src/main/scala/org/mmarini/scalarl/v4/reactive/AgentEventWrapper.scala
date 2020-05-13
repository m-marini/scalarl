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

package org.mmarini.scalarl.v4.reactive

import monix.reactive.Observable
import org.mmarini.scalarl.v4.Feedback
import org.mmarini.scalarl.v4.agents.{ActorCriticAgent, AgentEvent, PolicyActor, PriorityPlanner}
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j._

/**
 * Wrapper on step observable
 *
 * @param observable the observable
 */
class AgentEventWrapper(val observable: Observable[AgentEvent]) extends ObservableWrapper[AgentEvent] {
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
    val outs0 = agent.network.output(s0.signals)
    val outs1 = agent.network.output(s1.signals)
    val outs01 = agent1.network.output(s0.signals)
    val v0 = ActorCriticAgent.v(outs0)
    val v1 = ActorCriticAgent.v(outs1)
    val v01 = ActorCriticAgent.v(outs01)
    val (delta, vStar, _) = agent.computeDelta(v0, v1, reward)
    val deltaCritic = vStar.distance2(v0)
    val deltaCritic1 = vStar.distance2(v01)
    val actorsKpis = (for {
      ((a0, a1), action) <- agent.actors.zip(agent1.actors).zipWithIndex
    } yield {
      (a0, a1) match {
        case (actor: PolicyActor, actor1: PolicyActor) =>
          val pr = actor.preferences(outs0)
          val prStar = PolicyActor.computeActorLabel(pr, actions.getInt(action), actor.alpha, delta)
          val pr1 = actor1.preferences(outs01)
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
      deltaCritic * deltaCritic +:
        deltaCritic1 * deltaCritic1 +:
        (actorsKpis ++
          plannerKpis)).toArray)

    kpis
  }))
}