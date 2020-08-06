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

  def computeJ(labels: Array[INDArray], outputs0: Array[INDArray]): INDArray = {
    val l = hstack(labels: _ *)
    val o = hstack(outputs0: _ *)
    val result = pow(l.sub(o), 2).mean()
    result
  }

  /**
   *
   * @param event the agent event
   */
  private def toKpi(event: AgentEvent): INDArray = {
    event.agent match {
      case agent: ActorCriticAgent =>
        val map = event.map
        val map1 = event.map1

        val v0 = map("v0").asInstanceOf[INDArray]
        val v01 = map1("v0").asInstanceOf[INDArray]
        val vStar = map("v0*").asInstanceOf[INDArray]
        val avg = agent.avg
        val outputs0 = map("outputs0").asInstanceOf[Array[INDArray]]
        val labels = map("labels").asInstanceOf[Array[INDArray]]
        val outputs01 = map1("outputs0").asInstanceOf[Array[INDArray]]

        val j = computeJ(labels, outputs0)
        val j1 = computeJ(labels, outputs01)

        val actorsKpis = for {
          (actor, i) <- agent.actors.zipWithIndex
        } yield actor match {
          case a: PolicyActor =>
            val h = map(s"h($i)").asInstanceOf[INDArray]
            val hStar = map(s"h*($i)").asInstanceOf[INDArray]
            val h1 = map1(s"h($i)").asInstanceOf[INDArray]
            hstack(a.alpha, h, hStar, h1)
          case a: GaussianActor =>
            val mu = map(s"mu($i)").asInstanceOf[INDArray]
            val h = map(s"h($i)").asInstanceOf[INDArray]
            val muStar = map(s"mu*($i)").asInstanceOf[INDArray]
            val hStar = map(s"h*($i)").asInstanceOf[INDArray]
            val mu1 = map1(s"mu($i)").asInstanceOf[INDArray]
            val h1 = map1(s"h($i)").asInstanceOf[INDArray]
            hstack(a.eta.getColumn(0), mu, muStar, mu1,
              a.eta.getColumn(1L), h, hStar, h1)
          case _ => zeros(0)
        }

        val kpis = hstack((v0 +: vStar +: v01 +: avg +: j +: j1 +: actorsKpis).toArray: _ *)
        kpis
      case _ => zeros(0)
    }
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
    event.agent match {
      case agent: ActorCriticAgent =>
        val map = event.map
        val map1 = event.map1
        val v0 = map("v0").asInstanceOf[INDArray]
        val v01 = map1("v0").asInstanceOf[INDArray]
        val delta = map1("delta").asInstanceOf[INDArray]
        val vStar = map1("v0*").asInstanceOf[INDArray]
        logger.info("=============================")
        logger.info(" actions={}", event.feedback.actions)
        logger.info(" delta={}", delta)
        logger.info("v0={}, v0*={} v0'={}", v0, vStar, v01)

        for {
          (a0, i) <- agent.actors.zipWithIndex
        } yield {
          a0 match {
            case _: PolicyActor =>
              val h = map(s"h($i)").asInstanceOf[INDArray]
              val hStar = map(s"h*($i)").asInstanceOf[INDArray]
              val h1 = map1(s"h($i)").asInstanceOf[INDArray]
              logger.info(" h={},  h*={},  h'={}", h, hStar, h1)
            case _: GaussianActor =>
              val mu = map(s"mu($i)").asInstanceOf[INDArray]
              val h = map(s"h($i)").asInstanceOf[INDArray]
              val muStar = map(s"mu*($i)").asInstanceOf[INDArray]
              val hStar = map(s"h*$i)").asInstanceOf[INDArray]
              val mu1 = map1(s"mu($i)").asInstanceOf[INDArray]
              val h1 = map1(s"h($i)").asInstanceOf[INDArray]
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