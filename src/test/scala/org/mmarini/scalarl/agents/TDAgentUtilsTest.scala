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

package org.mmarini.scalarl.agents

import org.nd4j.linalg.factory.Nd4j
import org.scalatest.FunSpec
import org.scalatest.GivenWhenThen
import org.scalatest.Matchers

class TDAgentUtilsTest extends FunSpec with GivenWhenThen with Matchers {

  Nd4j.create()

  describe(s"a TDAgentUtils") {
    it("should return slice index for channels") {
      Given("a channel configuration")
      val conf = Array(1, 2, 3)

      When("compute slice index")
      val indexes = TDAgentUtils.sliceIdxFromChannels(conf)

      Then("slice indexes should be (0,0), (1,2), (3,5)")
      indexes should contain theSameElementsAs Array((0, 0), (1, 2), (3, 5))
    }

    it("should return slice index for a channel") {
      Given("a channel configuration")
      val conf = Array(8)

      When("compute slice index")
      val indexes = TDAgentUtils.sliceIdxFromChannels(conf)

      Then("slice indexes should be (0,7)")
      indexes should contain theSameElementsAs Array((0, 7))
    }

    it("should return action and value from policy") {
      Given("a channel configuration")
      val conf = Array(1, 2, 3)
      And("policy values")
      val policy = Nd4j.create(Array(
        1.0,
        2.0, 0.0,
        -2.0, 3.0, 0.0))
      And("full mask value")
      val mask = Nd4j.ones(6L)

      When("computes action and value")
      val (action, values) = TDAgentUtils.actionAndStatusValuesFromPolicy(policy, mask, conf)

      Then("should return action (1, 1, 0, 0, 1, 0")
      val expected = Nd4j.create(Array(
        1.0,
        1.0, 0.0,
        0.0, 1.0, 0.0))
      action shouldBe expected

      And("should return values (1, 2, 2, 3, 3, 3)")
      val expectedValues = Nd4j.create(Array(
        1.0,
        2.0, 2.0,
        3.0, 3.0, 3.0))
      values shouldBe expectedValues
    }

    it("should return action from policy with mask") {
      Given("a channel configuration")
      val conf = Array(1, 2, 3)
      And("policy values")
      val policy = Nd4j.create(Array(
        1.0,
        2.0, 0.0,
        -3.0, 4.0, 3.0))
      And("partial mask value")
      val mask = Nd4j.create(Array(
        1.0,
        1.0, 1.0,
        1.0, 0.0, 1.0))

      When("computes action")
      val (action, values) = TDAgentUtils.actionAndStatusValuesFromPolicy(policy, mask, conf)

      Then("should return (1, 1, 0, 0, 0, 1")
      val expected = Nd4j.create(Array(
        1.0,
        1.0, 0.0,
        0.0, 0.0, 1.0))
      action shouldBe expected

      And("should return values (1, 2, 2, 3, 3, 3)")
      val expectedValues = Nd4j.create(Array(
        1.0,
        2.0, 2.0,
        3.0, 3.0, 3.0))
      values shouldBe expectedValues
    }

    it("should return end state policy") {
      Given("a channel configuration")
      val conf = Array(1, 2, 3)

      When("computes end state policy")
      val policy = TDAgentUtils.endStatePolicy(conf)

      Then("should return zeros")
      policy shouldBe Nd4j.zeros(6)
    }

    it("should return fixed random action for fixed mask") {
      Given("a channel configuration")
      val conf = Array(1, 2, 3)

      And("fixed mask value")
      val mask = Nd4j.create(Array(1.0, 1.0, 0.0, 1.0, 0.0, 0.0))

      And("a deterministic random generator")
      val random = Nd4j.getRandomFactory().getNewRandomInstance(1234)

      When("computes random action")
      val policy = TDAgentUtils.randomAction(mask, conf)(random)

      Then("should return fixed action")
      val expected = Nd4j.create(Array(1.0, 1.0, 0.0, 1.0, 0.0, 0.0))
      policy shouldBe expected
    }

    it("should return random action") {
      Given("a channel configuration")
      val conf = Array(1, 2, 3)

      And("full mask value")
      val mask = Nd4j.ones(6L)

      And("a deterministic random generator")
      val random = Nd4j.getRandomFactory().getNewRandomInstance(1234)

      When("computes random action")
      val policy = TDAgentUtils.randomAction(mask, conf)(random)

      Then("should return action with random channels")
      val expected = Nd4j.create(Array(1.0, 1.0, 0.0, 0.0, 1.0, 0.0))
      policy shouldBe expected
    }

    it("should return fit policy") {
      Given("a channel configuration")
      val conf = Array(1, 2, 3)

      And("policy0 values")
      val policy0 = Nd4j.create(Array(
        1.0,
        2.0, 3.0,
        4.0, 5.0, 6.0))

      And("full mask value")
      val mask = Nd4j.ones(6L)

      And("policy1 values")
      val policy1 = Nd4j.create(Array(
        1.5,
        2.5, 3.5,
        4.5, 5.5, 6.5))

      And("an action")
      val action = Nd4j.create(Array(
        1.0,
        1.0, 0.0,
        0.0, 0.0, 1.0))

      And("a reward")
      val reward = 10.0

      And("a gamma reward discount")
      val gamma = 0.9

      And("a kappa hyper parameter")
      val kappa = 2

      When("computes fit policy")
      val policy = TDAgentUtils.bootstrapPolicy(policy0, mask, policy1, mask, action, reward, gamma, kappa, conf)

      /*
       * 1.0 + (10 + 1.5 * 0.9 - 1.0)/2 = 6,175
       * 3.0 + (10 + 3.5 * 0.9 - 3.0)/2 = 8,075
       * 3.0
       * 4.0
       * 5.0
       * 6.0 + (10 + 6.5 * 0.9 - 6.0)/2 = 10,925
       */
      Then("should return (6.175, 8.075, 3.0, 4.0, 5.0, 10.925")
      val expected = Nd4j.create(Array(
        6.175,
        8.075, 3.0,
        4.0, 5.0, 10.925))
      policy shouldBe expected
    }
  }
}
