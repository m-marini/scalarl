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

import org.mmarini.scalarl.v6.{Configuration, Feedback, INDArrayObservation}
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j.{ones, _}
import org.scalatest.mockito.MockitoSugar
import org.scalatest.{FunSpec, Matchers}

class PriorityPlannerJsonTest extends FunSpec with Matchers with MockitoSugar {

  val dt = 1

  create()
  private val jsonText1 =
    """
      |---
      |planner:
      |    planningSteps: 3
      |    minModelSize: 3
      |    maxModelSize: 10
      |    threshold: 0.1
      |    stateKey:
      |        type: Discrete
      |        ranges:
      |        - [-0.5, 1.5]
      |        noValues: [ 2 ]
      |    actionsKey:
      |        type: Discrete
      |        ranges:
      |        - [-0.5, 1.5]
      |        noValues: [ 2 ]
      |""".stripMargin

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
  describe(s"The PriorityPlanner build from $jsonText1") {
    val conf = Configuration.jsonFormString(jsonText1)
    val planner = PriorityPlanner.fromJson(conf.hcursor.downField("planner"))(1, 1).get

    it("should be empty") {
      planner.model shouldBe empty
    }

    it("should have threshold 0.1") {
      planner.threshold shouldBe 0.1
    }

    it("should compute key") {
      val s0 = state(0)
      val actions = action(0)
      val kso = planner.stateKeyGen(s0)
      val ka = planner.actionsKeyGen(actions)
      kso shouldBe ModelKey(Seq(0))
      ka shouldBe ModelKey(Seq(0))
    }

    it("should not enqueue low score entry") {
      val s0 = state(0)
      val s1 = state(0)
      val actions = action(0)
      val ks0 = planner.stateKeyGen(s0)
      val ka = planner.actionsKeyGen(actions)
      val f = feedback(dt, s0, actions, 0, s1)

      val model = planner.enqueue(((ks0, ka), (f, 0)))

      model shouldBe empty
    }

    it("should enqueue high score entry") {
      val s0 = state(0)
      val s1 = state(0)
      val actions = action(0)
      val ks0 = planner.stateKeyGen(s0)
      val ka = planner.actionsKeyGen(actions)
      val f = feedback(dt, s0, actions, 0, s1)

      val model = planner.enqueue(((ks0, ka), (f, 0.2)))

      model should contain((ks0, ka) -> (f, 0.2))
    }

    it("should delete low score entry") {
      val s0 = state(0)
      val s1 = state(0)
      val actions = action(0)
      val ks0 = planner.stateKeyGen(s0)
      val ka = planner.actionsKeyGen(actions)
      val f = feedback(dt, s0, actions, 0, s1)
      val model = planner.enqueue(((ks0, ka), (f, 0.2)))

      model should not be empty

      val p1 = planner.copy(model = model)
      val model1 = p1.enqueue(((ks0, ka), (f, 0.0)))

      model1 shouldBe empty
    }

    it("should replace high score entry") {
      val s0 = state(0)
      val s1 = state(0)
      val actions = action(0)
      val ks0 = planner.stateKeyGen(s0)
      val ka = planner.actionsKeyGen(actions)
      val f = feedback(dt, s0, actions, 0, s1)
      val model = planner.enqueue(((ks0, ka), (f, 0.2)))

      model should not be empty

      val p1 = planner.copy(model = model)
      val model1 = p1.enqueue(((ks0, ka), (f, 0.3)))

      model1 should contain((ks0, ka) -> (f, 0.3))
    }
  }
}
