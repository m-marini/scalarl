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

class ActorCriticAgentConfTest extends FunSpec with Matchers {

  create()

  describe("ActorCriticAgentConf") {
    val jsonText1 =
      """
        |---
        |agent:
        |    rewardDecay: 0.999
        |    valueDecay: 0.99
        |    rewardRange:
        |    -   [ -1, 1]
        |    actors:
        |    -   type: PolicyActor
        |        alpha: 0.1
        |        noValues: 5
        |        range:
        |        - [-1, 1]
        |        prefRange:
        |        - [-3, 3]
        |    -   type: PolicyActor
        |        alpha: 0.1
        |        noValues: 5
        |        range:
        |        - [-1, 1]
        |        prefRange:
        |        - [-3, 3]
        |    stateEncoder:
        |        type: Continuous
        |        ranges:
        |        - [-2, 3]
        |        - [-2, 4]
        |""".stripMargin
    it(s"should load from json $jsonText1") {
      val conf = Configuration.jsonFormString(jsonText1)
      val agentConf = ActorCriticAgentConf.fromJson(conf.hcursor.downField("agent"))(2, 2).get

      agentConf.netInputDimensions shouldBe 2
      agentConf.valueDecay shouldBe ones(1).muli(0.99)
      agentConf.rewardDecay shouldBe ones(1).muli(0.999)
      agentConf.stateEncode(zeros(2)) shouldBe create(Array(4.0 / 5 - 1, 4.0 / 6 - 1))
      agentConf.actors.length shouldBe 2
      agentConf.actors.head.noOutputs shouldBe 5
      agentConf.actors(1).noOutputs shouldBe 5
      agentConf.noOutputs shouldBe Seq(1, 5, 5)
    }

    val jsonText2 =
      """
        |---
        |agent:
        |    rewardDecay: 0.999
        |    valueDecay: 0.99
        |    rewardRange:
        |    -   [ -1, 1]
        |    actors:
        |    -   type: GaussianActor
        |        alphaMu: 0.1
        |        alphaSigma: 0.2
        |        muRange:
        |        - [-2, 2]
        |        sigmaRange:
        |        - [0.01, 0.1]
        |    -   type: GaussianActor
        |        alphaMu: 0.1
        |        alphaSigma: 0.2
        |        muRange:
        |        - [-2, 2]
        |        sigmaRange:
        |        - [0.01, 0.1]
        |    stateEncoder:
        |        type: Tiles
        |        ranges:
        |        - [0, 4]
        |        - [0, 2]
        |""".stripMargin
    it(s"should load from json $jsonText2") {
      val conf = Configuration.jsonFormString(jsonText2)
      val agentConf = ActorCriticAgentConf.fromJson(conf.hcursor.downField("agent"))(2, 2).get

      agentConf.netInputDimensions shouldBe 32
      agentConf.valueDecay shouldBe ones(1).muli(0.99)
      agentConf.rewardDecay shouldBe ones(1).muli(0.999)
      agentConf.stateEncode(zeros(2)) shouldBe create(Array(
        1.0, 0, 0, 0,
        1.0, 0, 0, 0,
        1.0, 0, 0, 0,
        1.0, 0, 0, 0,
        1.0, 0, 0, 0,
        1.0, 0, 0, 0,
        1.0, 0, 0, 0,
        1.0, 0, 0, 0
      ))
      agentConf.actors.length shouldBe 2
      agentConf.actors.head.noOutputs shouldBe 2
      agentConf.actors(1).noOutputs shouldBe 2
      agentConf.noOutputs shouldBe Seq(1, 2, 2)
    }
  }
}
