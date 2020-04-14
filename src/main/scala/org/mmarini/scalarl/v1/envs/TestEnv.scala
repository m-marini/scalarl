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

package org.mmarini.scalarl.v1.envs

import org.mmarini.scalarl.v1.{Action, Env, INDArrayObservation, Reward}
import org.nd4j.linalg.api.rng.Random
import org.nd4j.linalg.factory.Nd4j

/**
 *
 * @param time   the time
 * @param status the status
 * @param conf   the configuration
 */
case class TestEnv(time: Double, status: Int, conf: TestEnvConf) extends Env {

  /** Returns the [[org.mmarini.scalarl.v1.Observation]] for the environment */
  override lazy val observation: INDArrayObservation = {
    val signals = Nd4j.zeros(conf.numState)
    signals.putScalar(status, 1)
    val actions = 0L to 1L
    val endUp = status >= conf.numState - 1
    val o = INDArrayObservation(time = time,
      signals = signals,
      actions = actions,
      endUp = endUp)
    o
  }

  /**
   * Returns the environment simulator in reset status and the [Observation] of the reset status
   *
   * @param random the random generator
   */
  override def reset(random: Random): Env = copy(status = 0)

  /**
   * Computes the next status of environment executing an action.
   *
   * @param action the executing action
   * @param random the random generator
   * @return a n-uple with:
   *         - the environment in the next status,
   *         - the resulting observation,
   *         - the reward for the action,
   */
  override def change(action: Action, random: Random): (Env, Reward) = {
    val txs = conf.pi((status, action))
    val pi = random.nextDouble()
    val tx = txs.zipWithIndex.find(pi < _._1._1).get
    val result = tx match {
      case ((_, reward), status) => (copy(time = time + 1, status = status), reward)
    }
    result
  }

  /** Returns the number of signals */
  override def signalsSize: Int = conf.numState

  /** Returns the action channel configuration of the environment */
  override def actionsSize: Action = 2
}

/**
 *
 * @param numState number of states
 * @param pi       Map of (status, action)->(cumulative probability, reward)
 */
case class TestEnvConf(numState: Int, pi: Map[(Int, Int), Seq[(Double, Double)]])

/**
 *
 * @param numState number of state
 * @param pi       Map of (status, action, status + 1) -> (likely, reward),
 */
case class TestEnvConfBuilder(numState: Int, pi: Map[(Int, Int, Int), (Double, Double)]) {
  def numState(numState: Int): TestEnvConfBuilder = copy(numState = numState)

  /**
   * Returns the configuration builder
   *
   * @param s1 initial state
   * @param r  reward
   * @param s0 final state
   * @param a  action
   * @param p  likely
   */
  def p(s1: Int, r: Double, s0: Int, a: Int, p: Double): TestEnvConfBuilder = {
    require(s0 >= 0 && s0 < numState)
    require(a >= 0 && a <= 1)
    require(s1 >= 0 && s1 < numState)
    val pi1 = pi + ((s0, a, s1) -> (p, r))
    copy(pi = pi1)
  }

  /**
   *
   * @return
   */
  def build: TestEnvConf = {
    val pi1 = for {
      s <- 0 until numState
      a <- 0 to 1
    } yield {
      val tab = for {
        s1 <- 0 until numState
      } yield pi.getOrElse((s, a, s1), (0.0, 0.0))
      val tot = Math.max(tab.map(_._1).sum, 1e-3)
      val tab1 = tab.foldLeft(Seq[(Double, Double)]()) {
        case (Nil, (likely, reward: Reward)) => Seq((likely, reward))
        case (t, (likely, reward: Reward)) =>
          t :+ (t.last._1 + likely, reward)
      }
      (s, a) -> tab1.map(entry => (entry._1 / tot, entry._2))
    }
    TestEnvConf(numState, pi1.toMap)
  }
}

/**
 *
 */
object TestEnvConfBuilder {
  def apply(): TestEnvConfBuilder = TestEnvConfBuilder(0, Map())
}
