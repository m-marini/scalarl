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

import org.mmarini.scalarl.Feedback
import org.mmarini.scalarl.INDArrayObservation
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import org.scalatest.FunSpec
import org.scalatest.Matchers
import scala.math.abs
import org.mmarini.scalarl.INDArrayObservation
import org.scalatest.GivenWhenThen

class TraceTDAAgentTest extends FunSpec with GivenWhenThen with Matchers {
  val Outputs = 2
  val Inputs = 3
  val Eps = 1e-3

  val AvailableActions = Nd4j.create(Array(1.0, 1.0))

  val Feedbacks: Seq[Feedback] = Seq(
    createFeedback(
      in0 = Nd4j.create(Array(1.0, 0.0, 0.0)),
      action = 0,
      reward = -1,
      in1 = Nd4j.create(Array(0.0, 1.0, 0.0)),
      endUp = false),
    createFeedback(
      in0 = Nd4j.create(Array(0.0, 1.0, 0.0)),
      action = 0,
      reward = 1,
      in1 = Nd4j.create(Array(0.0, 0.0, 1.0)),
      endUp = true))

  def createFeedback(
    in0:    INDArray,
    action: Int,
    reward: Double,
    in1:    INDArray,
    endUp:  Boolean): Feedback =
    Feedback(
      s0 = INDArrayObservation(in0, AvailableActions),
      action = action,
      reward = reward,
      s1 = INDArrayObservation(in1, AvailableActions),
      endUp = endUp)

  def createAgent(): TDAAgent = {
    AgentBuilder()
      .numInputs(Inputs)
      .numActions(Outputs)
      .numHiddens(Outputs)
      .agentType(AgentType.TDAAgent)
      .build().asInstanceOf[TDAAgent]
  }

  describe(s"a TDQAgent") {
    it("should return highet q for greedy action") {
      Given("a TDQAgent")
      val agent = createAgent()

      And("an observation")
      val obs = Feedbacks(0).s0

      When("gets for greedyAction")
      val action = agent.greedyAction(obs)

      Then("t should return the action with the highest q")
      val q = agent.policy(obs)
      for { i <- 0L until q.length() if i != action } {
        q.getDouble(i) should be <= (q.getDouble(action.toLong))
      }
    }

    it("should fit from feedback") {
      Given("a TDAAgent")
      val agent = createAgent()

      And("a feedback")
      val feedback = Feedbacks(0)
      val Feedback(s0, action, reward, s1, _) = feedback

      When("get action value for (s0, action)")
      val beforeQ0 = agent.actionPolicy(s0, action)

      And("fits for the feedback")
      val (agent1, error) = agent.fit(feedback)

      And("gets the q of the fitted agent")
      val agent2 = agent1.asInstanceOf[TDAAgent]
      val afterQ0 = agent2.actionPolicy(s0, action)

      Then("value of action is nearer the expected value")
      val expectedQ0 = reward + agent.gamma * agent.v(s1)
      abs(expectedQ0 - afterQ0) should be <= (abs(expectedQ0 - beforeQ0))
    }
  }
}
