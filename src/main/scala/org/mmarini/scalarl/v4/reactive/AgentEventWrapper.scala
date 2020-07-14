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

import com.typesafe.scalalogging.LazyLogging
import monix.eval.Task
import monix.reactive.Observable
import org.mmarini.scalarl.v4.Feedback
import org.mmarini.scalarl.v4.agents.{ActorCriticAgent, AgentEvent, GaussianActor, PolicyActor}
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j._
import org.nd4j.linalg.ops.transforms.Transforms._

/**
 * Wrapper on step observable
 *
 * @param observable the observable
 */
class AgentEventWrapper(val observable: Observable[AgentEvent]) extends ObservableWrapper[AgentEvent] with LazyLogging {
  /**
   * Returns the kpis
   * The kpis consists of
   * - critic eta
   * - critic v
   * - critic v*
   * - critic v'
   * - average R
   * - actors
   *  - alpha
   *  - h
   *  - h*
   *  - h'
   */
  def kpis(): INDArrayWrapper = new INDArrayWrapper(observable.map(toKpi))

  /**
   *
   * @param event the agent event
   */
  private def toKpi(event: AgentEvent): INDArray = (event.agent0, event.agent1) match {
    case (agent: ActorCriticAgent, agent1: ActorCriticAgent) =>
      val Feedback(s0, actions, reward, s1) = event.feedback
      val outs0 = agent.network.output(s0.signals)
      val outs1 = agent.network.output(s1.signals)
      val outs01 = agent1.network.output(s0.signals)
      val v0 = agent.v(outs0)
      val v1 = agent.v(outs1)
      val v01 = agent1.v(outs01)
      val (delta, vStar, _) = agent.computeDelta(v0, v1, reward)
      val avg = agent1.avg
      val actorsKpis = for {
        ((a0, a1), _) <- agent.actors.zip(agent1.actors).zipWithIndex
      } yield {
        (a0, a1) match {
          case (actor: PolicyActor, actor1: PolicyActor) =>
            val h = actor.preferences(outs0)
            val oStar = actor.computeLabels(outs0, actions, delta)
            val hStar = actor.preferences(oStar)
            val h1 = actor1.preferences(outs01)
            hstack(actor.alpha, h, hStar, h1)
          case (actor: GaussianActor, actor1: GaussianActor) =>
            val (mu, h, muStar, hStar) = actor.muHStar(outs0, actions, delta)
            val (mu1, h1, _) = actor1.muHSigma(outs0)
            hstack(actor.eta.getColumn(0), mu, muStar, mu1,
              actor.eta.getColumn(1L), h, hStar, h1)
          case _ => zeros(0)
        }
      }
      val kpis = hstack((v0 +: vStar +: v01 +: avg +: actorsKpis).toArray: _ *)
      kpis
    case _ => zeros(0)
  }

  /**
   *
   */
  def logKpi(): AgentEventWrapper = {
    val ob1 = observable.doOnNext(event => Task.eval {
      logEvent(event)
    })
    new AgentEventWrapper(ob1)
  }

  /**
   *
   * @param event the agent event
   */
  private def logEvent(event: AgentEvent) {
    (event.agent0, event.agent1) match {
      case (agent: ActorCriticAgent, agent1: ActorCriticAgent) =>
        val Feedback(s0, actions, reward, s1) = event.feedback
        val outs0 = agent.network.output(s0.signals)
        val outs1 = agent.network.output(s1.signals)
        val outs01 = agent1.network.output(s0.signals)
        val v0 = agent.v(outs0)
        val v1 = agent.v(outs1)
        val v01 = agent1.v(outs01)
        val (delta, vStar, _) = agent.computeDelta(v0, v1, reward)
        val avg = agent1.avg
        logger.info("=============================")
        logger.info(" actions={}", actions)
        logger.info(" delta={}", delta)
        logger.info("v0={}, v0*={} v0'={}", v0, vStar, v01)

        for {
          ((a0, a1), _) <- agent.actors.zip(agent1.actors).zipWithIndex
        } yield {
          (a0, a1) match {
            case (actor: PolicyActor, actor1: PolicyActor) =>
            //val h = actor.preferences(outs0)
            //val oStar = actor.computeLabels(outs0, actions, delta)
            //val hStar = actor.preferences(oStar)
            //val h1 = actor1.preferences(outs01)
            case (actor: GaussianActor, actor1: GaussianActor) =>
              val (mu, h, muStar, hStar) = actor.muHStar(outs0, actions, delta)
              val (mu1, h1, _) = actor1.muHSigma(outs01)
              logger.info("mu={}, mu*={}, mu'={}", mu, muStar, mu1)
              logger.info(" h={},  h*={},  h'={}", h, hStar, h1)
              logger.info("sg={}, sg*={}, sg'={}", exp(h), exp(hStar), exp(h1))
            case _ =>
              logger.info("No match")
          }
        }
    }
  }
}