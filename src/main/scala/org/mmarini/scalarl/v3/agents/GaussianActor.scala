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

import java.io.File

import com.typesafe.scalalogging.LazyLogging
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.util.ModelSerializer
import org.mmarini.scalarl.v3._
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.api.rng.Random
import org.nd4j.linalg.factory.Nd4j._
import org.nd4j.linalg.ops.transforms.Transforms._

/**
 * Actor critic agent
 *
 * @param dimension the dimension index
 * @param actor     the actor network for mu
 * @param alpha     the alpha parameter
 */
case class GaussianActor(dimension: Int,
                         actor: MultiLayerNetwork,
                         alpha: INDArray) extends Actor with LazyLogging {

  /**
   * Returns the new agent and the chosen action.
   * Chooses the action to be executed to the environment.
   *
   * @param observation the observation of environment
   * @param random      the random generator
   */
  override def chooseAction(observation: Observation, random: Random): INDArray = {
    val (mu, _, sigma) = muHSigma(observation)
    val action = sigma.mul(random.nextGaussian()).addi(mu)
    action
  }

  /**
   * Returns mu, h, sigma
   *
   * @param s0 the state observation
   */
  def muHSigma(s0: Observation): (INDArray, INDArray, INDArray) = GaussianActor.muHSigma(actor.output(s0.signals))

  /**
   * Returns the fit agent by optimizing its strategy policy and the score
   *
   * @param feedback the feedback from the last step
   * @param random   the random generator
   */
  override def fit(feedback: Feedback, delta: INDArray, random: Random): Actor = {
    val Feedback(s0, actions, reward, s1) = feedback
    val action = actions.getColumn(dimension)

    // Actor update
    val (mu, h, sigma) = muHSigma(s0)

    val (mu1, h1) = GaussianActor.computeActorTarget(action, alpha, delta, mu, h, sigma)

    val actorLabels = hstack(mu1, h1)
    val newActor = actor.clone()
    newActor.fit(s0.signals, actorLabels)

    val newActpr = copy(actor = newActor)
    newActpr
  }

  /**
   * Writes the agent status to file
   *
   * @param file the model folder
   * @return the agents
   */
  override def writeModel(file: File): Actor = {
    ModelSerializer.writeModel(actor, new File(file, s"actor-$dimension.zip"), false)
    this
  }
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
