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

package org.mmarini.scalarl.v3.agents

import org.mmarini.scalarl.v3.{Agent, Feedback, INDArrayObservation}
import org.mockito.Mockito._
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j.{ones, _}
import org.scalatest.mockito.MockitoSugar
import org.scalatest.{FunSpec, Matchers}

class PriorityPlannerTest1 extends FunSpec with Matchers with MockitoSugar {

  val dt = 1

  create()

  def state(n: Int): INDArray = ones(1).muli(n)

  def action(n: Int): INDArray = ones(1).muli(n)

  def obs(time: Double, s: INDArray): INDArrayObservation =
    INDArrayObservation(time = ones(1).muli(time), signals = s)

  def feedback(time: Double, s0: INDArray, a: INDArray, r: Double, s1: INDArray): Feedback = Feedback(
    s0 = obs(time, s0),
    actions = a,
    reward = ones(1).muli(r),
    s1 = obs(time + dt, s1)
  )

  describe("A PriorityPlanner") {
    val stateKeyGen = (x: INDArray) => x
    val actionsKeyGen = (x: INDArray) => x
    val fo = Ordering.by((p: ((INDArray, INDArray), Feedback)) => p match {
      case (_, f) => f.s0.time.getDouble(0L)
    }).reverse

    val model = Model[(INDArray, INDArray), Feedback](minModelSize = 1,
      maxModelSize = 10,
      data = Map(),
      ordering = fo)

    val queue = PriorityQueue[(INDArray, INDArray)](threshold = 0.1,
      queue = Map())

    val planner = PriorityPlanner(stateKeyGen = stateKeyGen,
      actionsKeyGen = actionsKeyGen,
      planningSteps = 2,
      model = model,
      queue = queue)

    val s0 = state(0)
    val s1 = state(1)
    val s2 = state(2)
    val a0 = action(0)

    describe("when planning backward twice") {
      val f0 = feedback(0, s0, a0, 1, s1)
      val f1 = feedback(1, s1, a0, 1, s2)
      val random = getRandomFactory.getNewRandomInstance(1234)

      val agent0 = mock[Agent]
      val agent1 = mock[Agent]

      when(agent0.score(f0)).thenReturn(ones(1).muli(1))
      when(agent0.score(f1)).thenReturn(ones(1).muli(2))
      val r0 = (agent1, ones(1).muli(0.75))
      when(agent0.directLearn(f1, random)).thenReturn(r0)

      when(agent1.score(f0)).thenReturn(ones(1).muli(7.0 / 8))
      val r1 = (agent1, ones(1).muli(0.5))
      when(agent1.directLearn(f0, random)).thenReturn(r1)

      val p1 = planner.learn(feedback = f0, agent = agent0).
        learn(f1, agent0).asInstanceOf[PriorityPlanner[INDArray, INDArray]]

      it("should contain initial queue with 2 elements") {
        p1.queue.queue should have size (2)
        p1.queue.queue should contain((s0, a0) -> 1.0)
        p1.queue.queue should contain((s1, a0) -> 2.0)
      }

      val (_, p2) = p1.plan(agent0, random)
      val p3 = p2.asInstanceOf[PriorityPlanner[INDArray, INDArray]]

      it("should contain queue with 2 elements") {
        p3.queue.queue should have size (2)
        p3.queue.queue should contain((s0, a0) -> 0.5)
        p3.queue.queue should contain((s1, a0) -> 0.75)
      }

      it("should call directLearn to agent") {
        verify(agent0).directLearn(f1, random)
        verify(agent1).directLearn(f0, random)
      }
    }
  }
}
