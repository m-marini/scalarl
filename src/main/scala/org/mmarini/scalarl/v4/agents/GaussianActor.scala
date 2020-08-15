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
import io.circe.ACursor
import org.mmarini.scalarl.v4.Utils._
import org.mmarini.scalarl.v4.envs.Configuration._
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.api.rng.Random
import org.nd4j.linalg.factory.Nd4j._
import org.nd4j.linalg.ops.transforms.Transforms._

import scala.util.Try

/**
 * Gaussian Actor
 *
 * @param dimension   the dimension index
 * @param eta         the eta (mu, sigma) parameters
 * @param denormalize the output denormalizer function
 * @param normalize   the output normalizer function
 */
case class GaussianActor(dimension: Int,
                         eta: INDArray,
                         denormalize: INDArray => INDArray,
                         normalize: INDArray => INDArray) extends Actor with LazyLogging {
  /**
   * Returns the actor labels
   *
   * @param outputs the outputs
   * @param actions the actions
   * @param delta   the td error
   */
  override def computeLabels(outputs: Array[INDArray],
                             actions: INDArray,
                             delta: INDArray): Map[String, Any] = {
    val (mu, h, sigma) = muHSigma(outputs)
    val action = actions.getColumn(dimension)
    val sigma2 = sigma.mul(sigma)
    val deltaAction = action.sub(mu)
    val ratio = deltaAction.div(sigma)
    // deltaMu = 2 (action - mu) / sigma^2 delta
    val deltaMu = deltaAction.
      div(sigma2).
      muli(delta).
      muli(2).
      muli(eta.getColumn(0))
    // deltaH = (2 (action - mu)^2 / sigma^2) - 1) delta
    val deltaH = ratio.mul(ratio).muli(2).subi(1).muli(delta).muli(eta.getColumn(1))

    val muStar = mu.add(deltaMu)
    val hStar = h.add(deltaH)
    val labels = normalize(hstack(muStar, hStar))
    val clipLabeles = clip(labels, -1, 1)
    Map(s"mu($dimension)" -> mu,
      s"h($dimension)" -> h,
      s"sigma($dimension)" -> sigma,
      s"deltaMu($dimension)" -> deltaMu,
      s"deltaH($dimension)" -> deltaH,
      s"mu*($dimension)" -> muStar,
      s"h*($dimension)" -> hStar,
      s"labels($dimension)" -> clipLabeles
    )
  }

  /**
   * Returns mu, h, sigma
   *
   * @param outputs the output of actor network
   */
  def muHSigma(outputs: Array[INDArray]): (INDArray, INDArray, INDArray) = {
    val denorm = denormalize(outputs(dimension + 1))
    val mu = denorm.getColumn(0)
    val h = denorm.getColumn(1)
    val sigma = exp(h)
    (mu, h, sigma)
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

  /** Returns the number of outputs */
  override def noOutputs: Int = 2
}

object GaussianActor {

  /**
   * Returns the discrete action agent
   *
   * @param conf      the configuration element
   * @param dimension the dimension index
   * @param noInputs  the number of inputs
   * @param modelPath the path of model to load
   */
  def fromJson(conf: ACursor)(dimension: Int,
                              noInputs: Int,
                              modelPath: Option[String]): Try[GaussianActor] = for {
    alphaMu <- conf.get[Double]("alphaMu").toTry
    alphaSigma <- conf.get[Double]("alphaSigma").toTry
    muRange <- rangesFromJson(conf.downField("muRange"))(1)
    sigmaRange <- rangesFromJson(conf.downField("sigmaRange"))(1).flatMap { range =>
      Try {
        require(range.getDouble(0L, 0L) > 0, s"sigmaRange must be positive")
        range
      }
    }}
    yield {
      val range = hstack(muRange, log(sigmaRange))
      val denorm = denormalize(range)
      val norm = normalize(range)
      GaussianActor(
        dimension = dimension,
        eta = create(Array(alphaMu, alphaSigma)),
        denormalize = denorm,
        normalize = norm)
    }
}