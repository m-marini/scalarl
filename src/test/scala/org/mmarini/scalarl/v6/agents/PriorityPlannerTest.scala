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

package org.mmarini.scalarl.v6.agents

import org.mmarini.scalarl.v6.{Agent, Feedback, INDArrayObservation}
import org.mockito.ArgumentMatchers._
import org.mockito.Mockito._
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j.{ones, _}
import org.scalatest.mockito.MockitoSugar
import org.scalatest.{FunSpec, Matchers}

class PriorityPlannerTest extends FunSpec with Matchers with MockitoSugar {

  val dt = 1

  create()

  def state(n: Int): INDArray = ones(1).muli(n)

  def action(n: Int): INDArray = ones(1).muli(n)

  def feedback(time: Double, s0: INDArray, a: INDArray, r: Double, s1: INDArray): Feedback = Feedback(
    s0 = obs(time, s0),
    actions = a,
    reward = ones(1).muli(r),
    s1 = obs(time + dt, s1)
  )

  def obs(time: Double, s: INDArray): INDArrayObservation =
    INDArrayObservation(time = ones(1).muli(time), signals = s)

  describe("A PriorityPlanner") {
    val stateKeyGen = (x: INDArray) => x
    val actionsKeyGen = (x: INDArray) => x

    val planner = PriorityPlanner[INDArray, INDArray](stateKeyGen = stateKeyGen,
      actionsKeyGen = actionsKeyGen,
      planningSteps = 1,
      minModelSize = 1,
      maxModelSize = 10,
      threshold = 0.1,
      model = Map())

    val s0 = state(0)
    val s1 = state(1)
    val s2 = state(2)
    val a0 = action(0)

    describe("when learning data with low score") {
      val f0 = feedback(0, s0, a0, 1, s1)
      val agent = mock[Agent]
      when(agent.score(any())).thenReturn(zeros(1))
      val r = planner.learn(feedback = f0, agent = agent).asInstanceOf[PriorityPlanner[INDArray, INDArray]]

      it("should learn model") {
        /*
                r.model.get((s0, a0)) should matchPattern {
                  case Some((f, _)) if f == f0 =>
                }
        */
        r.model.get((s0, a0)) shouldBe None
      }
      it("should call score to agent") {
        verify(agent).score(any())
      }
    }

    describe("when learning data with high score") {
      val f0 = feedback(0, s0, a0, 1, s1)
      val agent = mock[Agent]
      when(agent.score(any())).thenReturn(ones(1))
      val r = planner.learn(feedback = f0, agent = agent).asInstanceOf[PriorityPlanner[INDArray, INDArray]]

      it("should learn model") {
        r.model.get((s0, a0)) should matchPattern {
          case Some((f, _)) if f == f0 =>
        }
      }
      it("should call score to agent") {
        verify(agent).score(any())
      }
    }

    describe("when learning data with mix score") {
      val f0 = feedback(0, s0, a0, 1, s1)
      val f1 = feedback(1, s1, a0, 1, s2)
      val agent = mock[Agent]
      when(agent.score(f0)).thenReturn(zeros(1))
      when(agent.score(f1)).thenReturn(ones(1))
      val r = planner.learn(feedback = f0, agent = agent).
        learn(f1, agent).asInstanceOf[PriorityPlanner[INDArray, INDArray]]

      it("should learn model") {
        /*
                r.model.get((s0, a0)) should matchPattern {
                  case Some((f, _)) if f == f0 =>
                }
        */
        r.model.get((s0, a0)) shouldBe None
        r.model.get((s1, a0)) should matchPattern {
          case Some((f, _)) if f == f1 =>
        }
      }
      it("should call score to agent") {
        verify(agent).score(f0)
        verify(agent).score(f1)
      }
    }

    describe("when planning no backward") {
      val f0 = feedback(0, s0, a0, 1, s1)
      val f1 = feedback(1, s1, a0, 1, s2)
      val random = getRandomFactory.getNewRandomInstance(1234)

      val agent1 = mock[Agent]
      val agent0 = mock[Agent]

      when(agent0.score(f0)).thenReturn(ones(1).muli(2))
      when(agent0.score(f1)).thenReturn(ones(1).muli(1))
      val r0 = (agent1, ones(1), ones(1).muli(0.75))
      when(agent0.directLearn(f0, random)).thenReturn(r0)

      when(agent1.score(f0)).thenReturn(ones(1).muli(0.25))
      when(agent1.score(f1)).thenReturn(ones(1).muli(1))
      val r1 = (agent1, ones(1), ones(1).muli(0.5))
      when(agent1.directLearn(f1, random)).thenReturn(r1)

      val p1 = planner.learn(feedback = f0, agent = agent0).
        learn(f1, agent0).asInstanceOf[PriorityPlanner[INDArray, INDArray]]

      p1.plan(agent0, random)

      it("should call directLearn to agent") {
        verify(agent0).directLearn(f0, random)
      }
    }

    describe("when planning backward") {
      val f0 = feedback(0, s0, a0, 1, s1)
      val f1 = feedback(1, s1, a0, 1, s2)
      val random = getRandomFactory.getNewRandomInstance(1234)

      val agent0 = mock[Agent]
      val agent1 = mock[Agent]

      when(agent0.score(f0)).thenReturn(ones(1).muli(1))
      when(agent0.score(f1)).thenReturn(ones(1).muli(2))
      val r0 = (agent1, ones(1), ones(1).muli(1))
      when(agent0.directLearn(f1, random)).thenReturn(r0)

      when(agent1.score(f0)).thenReturn(ones(1).muli(0.5))

      val p1 = planner.learn(feedback = f0, agent = agent0).
        learn(f1, agent0).asInstanceOf[PriorityPlanner[INDArray, INDArray]]

      val (_, p2) = p1.plan(agent0, random)
      p2.asInstanceOf[PriorityPlanner[INDArray, INDArray]]

      it("should call directLearn to agent") {
        verify(agent0).directLearn(f1, random)
      }
    }

    describe("when planning backward twice") {
      val f0 = feedback(0, s0, a0, 1, s1)
      val f1 = feedback(1, s1, a0, 1, s2)
      val random = getRandomFactory.getNewRandomInstance(1234)

      val agent0 = mock[Agent]
      val agent1 = mock[Agent]

      when(agent0.score(f0)).thenReturn(ones(1).muli(1))
      when(agent0.score(f1)).thenReturn(ones(1).muli(2))
      val r0 = (agent1, ones(1), ones(1).muli(0.75))
      when(agent0.directLearn(f1, random)).thenReturn(r0)

      when(agent1.score(f0)).thenReturn(ones(1).muli(7.0 / 8))
      val r1 = (agent1, ones(1), ones(1).muli(0.5))
      when(agent1.directLearn(f0, random)).thenReturn(r1)

      val p1 = planner.learn(feedback = f0, agent = agent0).
        learn(f1, agent0).asInstanceOf[PriorityPlanner[INDArray, INDArray]]

      val (_, p2) = p1.plan(agent0, random)

      it("should call directLearn to agent") {
        verify(agent0).directLearn(f1, random)
      }

      describe("when planning second time") {
        p2.plan(agent1, random)

        it("should call directLearn to agent") {
          verify(agent1).directLearn(f0, random)
        }
      }
    }
  }
}
