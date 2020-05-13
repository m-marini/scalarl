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

package org.mmarini.scalarl.v4.agents

import com.typesafe.scalalogging.LazyLogging
import org.mmarini.scalarl.v4._
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.api.rng.Random
import org.nd4j.linalg.factory.Nd4j.hstack
import org.nd4j.linalg.ops.transforms.Transforms._

/**
 * Gaussian Actor
 *
 * @param dimension the dimension index
 * @param alpha     the alpha parameter
 */
case class GaussianActor(dimension: Int,
                         alpha: INDArray) extends Actor with LazyLogging {
  /**
   * Returns the actor labels
   *
   * @param outputs the outputs
   * @param actions the actions
   * @param delta   the td error
   * @param random  the random generator
   */
  override def computeLabels(outputs: Array[INDArray], actions: INDArray, delta: INDArray, random: Random): INDArray = {
    val (mu, h, sigma) = muHSigma(outputs)
    val (mu1, h1) = GaussianActor.computeActorTarget(actions.getColumn(dimension), alpha, delta, mu, h, sigma)
    val actorLabels = hstack(mu1, h1)
    actorLabels
  }

  /**
   * Returns the action choosen by the actor
   *
   * @param outputs the network outputs
   * @param random  the random generator
   */
  override def chooseAction(outputs: Array[INDArray], random: Random): INDArray = {
    val (mu, _, sigma) = muHSigma(outputs)
    val action = sigma.mul(random.nextGaussian()).addi(mu)
    action
  }

  /**
   * Returns mu, h, sigma
   *
   * @param outputs the output of actor network
   */
  def muHSigma(outputs: Array[INDArray]): (INDArray, INDArray, INDArray) =
    GaussianActor.muHSigma(outputs(dimension + 1))

  /** Returns the number of outputs */
  override def noOutputs: Int = 2
}

/**
 *
 */
object GaussianActor {
  val MuRange = 1e100
  val HRange = 100.0

  /**
   * Returns mu, h, sigma
   *
   * @param out the output of actor network
   */
  def muHSigma(out: INDArray): (INDArray, INDArray, INDArray) = {
    val mu = Utils.clip(out.getColumn(0), -MuRange, MuRange)
    val h = Utils.clip(out.getColumn(1), -HRange, HRange)
    val sigma = exp(h)
    (mu, h, sigma)
  }

  /**
   * Returns the target mu, h
   *
   * @param action the action
   * @param alpha  the alpha parameter
   * @param delta  the TD Error
   * @param mu     the mu
   * @param h      the h sigma
   * @param sigma  the sigma
   */
  def computeActorTarget(action: INDArray,
                         alpha: INDArray,
                         delta: INDArray,
                         mu: INDArray,
                         h: INDArray,
                         sigma: INDArray): (INDArray, INDArray) = {
    val sigma2 = sigma.mul(sigma)
    val deltaAction = action.sub(mu)
    val ratio = deltaAction.div(sigma)
    // deltaMu = 2 (action - mu) / sigma^2 delta
    val deltaMu = deltaAction.div(sigma2).muli(delta).muli(2)
    // deltaH = (2 (action - mu)^2 / sigma^2) - 1) delta
    val deltaH = ratio.mul(ratio).muli(2).subi(1).muli(delta).muli(alpha)
    val mu1 = Utils.clip(mu.add(deltaMu), -MuRange, MuRange)
    val h1 = Utils.clip(h.add(deltaH), -HRange, HRange)
    (mu1, h1)
  }
}
