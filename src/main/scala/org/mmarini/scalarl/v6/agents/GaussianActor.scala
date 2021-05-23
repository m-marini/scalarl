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

import com.typesafe.scalalogging.LazyLogging
import io.circe.ACursor
import org.mmarini.scalarl.v6.Configuration._
import org.mmarini.scalarl.v6.Utils._
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.api.rng.Random
import org.nd4j.linalg.factory.Nd4j._
import org.nd4j.linalg.ops.transforms.Transforms._

import scala.util.Try

/**
 * Gaussian Actor
 *
 * @param dimension   the dimension index
 * @param epsilonMu   the epsilon mu parameter
 * @param epsilonH    the epsilon H parameter
 * @param alphaDecay  the alpha decay parameter
 * @param denormalize the output denormalize function
 * @param normalize   the output normalizer function
 */
case class GaussianActor(dimension: Int,
                         epsilonMu: INDArray,
                         epsilonH: INDArray,
                         alphaDecay: Double,
                         denormalize: INDArray => INDArray,
                         normalize: INDArray => INDArray) extends Actor with LazyLogging {

  import GaussianActor._

  /**
   * Returns the actor labels
   *
   * @param outputs the outputs
   * @param actions the actions
   * @param delta   the td error
   * @param alpha   the alpha parameters
   */
  override def computeLabels(outputs: Array[INDArray],
                             actions: INDArray,
                             delta: INDArray,
                             alpha: INDArray): Map[String, Any] = {
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
      muli(alpha.getColumn(0))
    // deltaH = (2 (action - mu)^2 / sigma^2) - 1) delta
    val deltaH = ratio.mul(ratio).muli(2).subi(1).muli(delta).muli(alpha.getColumn(1))

    val muStar = mu.add(deltaMu)
    val hStar = h.add(deltaH)
    val labels = normalize(hstack(muStar, hStar))

    // Compute alpha
    val deltaMuRMS = abs(deltaMu)
    val alphaMu1 = if (deltaMuRMS.getDouble(0L) > MinEpsilonH) {
      epsilonMu.div(deltaMuRMS)
    } else {
      alpha.getColumn(0)
    }
    val deltaHRMS = abs(deltaH)
    val alphaH1 = if (deltaHRMS.getDouble(0L) > MinEpsilonH) {
      epsilonMu.div(deltaHRMS)
    } else {
      alpha.getColumn(1)
    }

    val alpha1 = hstack(alphaMu1, alphaH1)
    val alphaStar = alpha.mul(alphaDecay).addi(alpha1.mul(1 - alphaDecay))

    Map(s"mu($dimension)" -> mu,
      s"h($dimension)" -> h,
      s"sigma($dimension)" -> sigma,
      s"deltaMu($dimension)" -> deltaMu,
      s"deltaH($dimension)" -> deltaH,
      s"mu*($dimension)" -> muStar,
      s"h*($dimension)" -> hStar,
      s"labels($dimension)" -> labels,
      s"alpha*(0)" -> alphaStar
    )
  }

  /**
   * Returns the action chosen by the actor
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
  def muHSigma(outputs: Array[INDArray]): (INDArray, INDArray, INDArray) = {
    val fDenormalize = denormalize(outputs(dimension + 1))
    val mu = fDenormalize.getColumn(0)
    val h = fDenormalize.getColumn(1)
    val sigma = exp(h)
    (mu, h, sigma)
  }

  /** Returns the number of outputs */
  override def noOutputs: Int = 2
}

object GaussianActor {

  val MinEpsilonH = 1e-3


  /**
   * Returns the gaussian actor and the alpha parameters
   *
   * @param conf      the configuration element
   * @param dimension the dimension index
   */
  def fromJson(conf: ACursor)(dimension: Int): Try[(GaussianActor, INDArray)] = for {
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
    val alphaDecay = conf.get[Double]("alphaDecay").getOrElse(1.0)
    val epsilon = conf.get[Double]("epsilon").getOrElse(0.1)
    (createActor(
      dimension = dimension,
      alphaMu = alphaMu,
      alphaSigma = alphaSigma,
      muRange = muRange,
      sigmaRange = sigmaRange,
      alphaDecay = alphaDecay,
      epsilon = epsilon
    ), create(Array(alphaMu, alphaSigma)))
  }

  /**
   * Return a Gaussian actor
   *
   * @param dimension  the dimension index
   * @param alphaMu    the alpha mu parameter
   * @param alphaSigma the alpha sigma parameter
   * @param epsilon    the epsilon parameter
   * @param alphaDecay the alpha decay parameter
   * @param muRange    the mu range
   * @param sigmaRange the sigma range
   */
  def createActor(dimension: Int,
                  alphaMu: Double,
                  alphaSigma: Double,
                  epsilon: Double,
                  alphaDecay: Double,
                  muRange: INDArray,
                  sigmaRange: INDArray): GaussianActor = {
    val range = hstack(muRange, log(sigmaRange))
    val fDenormalize = clipAndDenormalize(range)
    val norm = clipAndNormalize(range)
    val epsilonRange = range.getRow(1L).sub(range.getRow(0L)).muli(epsilon)
    GaussianActor(
      dimension = dimension,
      epsilonMu = epsilonRange.getColumn(0L),
      epsilonH = epsilonRange.getColumn(1L),
      alphaDecay = alphaDecay,
      denormalize = fDenormalize,
      normalize = norm)
  }
}