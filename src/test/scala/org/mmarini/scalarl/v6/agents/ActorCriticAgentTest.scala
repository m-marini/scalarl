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

import org.mmarini.scalarl.v6.Configuration
import org.nd4j.linalg.factory.Nd4j._
import org.scalatest.{FunSpec, Matchers}

class ActorCriticAgentTest extends FunSpec with Matchers {
  val V0 = 1.1
  val V1 = 3.0
  val Reward = 2.0
  val Average = 1.3
  val ValueDecay = 0.9
  val RewardDecay = 0.8

  create()

  describe("ActorCriticAgent") {
    it("should compute delta") {
      val v0 = ones(1).muli(V0)
      val v1 = ones(1).muli(V1)
      val reward = ones(1).muli(Reward)
      val avg = ones(1).muli(Average)
      val valueDecay = ones(1).muli(ValueDecay)
      val rewardDecay = ones(1).muli(RewardDecay)

      val (delta, newV0, newAvg) = ActorCriticAgent.computeDelta(v0, v1, reward, avg, valueDecay, rewardDecay)

      val ExpectedV0 = ValueDecay * (V1 + Reward - Average) + (1 - ValueDecay) * Average
      val ExpectedDelta = ExpectedV0 - V0
      val ExpectedAverage = RewardDecay * Average + (1 - RewardDecay) * Reward

      newV0 shouldBe ones(1).muli(ExpectedV0)
      delta shouldBe ones(1).muli(ExpectedDelta)
      newAvg shouldBe ones(1).muli(ExpectedAverage)
    }

    val jsonText =
      """
        |---
        |agent:
        |    type: ActorCritic
        |    avgReward: 0.1
        |    rewardDecay: 0.999
        |    valueDecay: 0.99
        |    rewardRange:
        |    -   [ -1, 1]
        |    actors:
        |    -  type: PolicyActor
        |       alpha: 0.1
        |       noValues: 5
        |       range:
        |       - [-1, 1]
        |       prefRange:
        |       - [-3, 3]
        |    -  type: PolicyActor
        |       alpha: 0.1
        |       noValues: 5
        |       range:
        |       - [-1, 1]
        |       prefRange:
        |       - [-3, 3]
        |    stateEncoder:
        |        type: Continuous
        |        ranges:
        |        - [-2, 3]
        |        - [-2, 4]
        |    network:
        |        seed: 1234
        |        learningRate: 30e-3
        |        activation: SOFTPLUS
        |        numHidden: []
        |        shortcuts: []
        |        updater: Sgd
        |        maxAbsGradient: 100
        |        maxAbsParameter: 10e3
        |        dropOut: 0.8
        |""".stripMargin
    it(s"should load from yaml $jsonText") {
      val conf = Configuration.jsonFormString(jsonText)

      val agent = AgentBuilder.fromJson(conf.hcursor.downField("agent"))(2, 2).get

      agent.avg shouldBe ones().muli(0.1)
      agent.planner shouldBe empty
    }
  }
}
