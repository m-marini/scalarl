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
import org.mmarini.scalarl.v6.Utils.{encode, _}
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.api.rng.Random
import org.nd4j.linalg.ops.transforms.Transforms._

import scala.util.Try

/**
 * Actor critic agent
 *
 * @param dimension   the dimension index
 * @param noOutputs   the number of value for action
 * @param epsilonH    the epsilonH parameter
 * @param alphaDecay  the alphaDecay
 * @param denormalize the outputs denormalize function to preference
 * @param normalize   the preference normalize function to outputs
 * @param decode      the function that decodes the action index to agent action value
 * @param encode      the function that encodes the agent action value to output feature vector
 */
case class PolicyActor(dimension: Int,
                       noOutputs: Int,
                       epsilonH: INDArray,
                       alphaDecay: Double,
                       denormalize: INDArray => INDArray,
                       normalize: INDArray => INDArray,
                       decode: Int => INDArray,
                       encode: INDArray => INDArray) extends Actor with LazyLogging {

  import PolicyActor._

  /**
   * Returns the action chosen by the actor
   *
   * @param outputs the network outputs
   * @param random  the random generator
   */
  override def chooseAction(outputs: Array[INDArray], random: Random): INDArray = {
    val pi = softmax(preferences(outputs))
    val actionIndex = randomInt(pi)(random)
    val action = decode(actionIndex)
    action
  }

  /**
   * Returns the preferences
   *
   * @param outputs the outputs
   */
  def preferences(outputs: Array[INDArray]): INDArray = {
    val outs = outputs(dimension + 1)
    val result = denormalize(outs)
    result
  }

  /**
   * Returns the actor labels
   *
   * @param outputs the outputs
   * @param actions the actions
   * @param delta   the td error
   */
  override def computeLabels(outputs: Array[INDArray],
                             actions: INDArray,
                             delta: INDArray,
                             alpha: INDArray): Map[String, INDArray] = {
    val h = preferences(outputs)
    val pi = softmax(h)
    val features = encode(actions.getScalar(dimension.toLong))
    val z = features.subi(pi)
    // deltaH = z * delta * alpha
    val deltaH = z.mul(delta).muli(alpha)
    // hStar = h + deltaH
    val hStar = h.add(deltaH)
    val actorLabels = normalize(hStar)
    val deltaHRMS = sqrt(pow(deltaH, 2).mean())
    val alphaStar = if (deltaHRMS.getDouble(0L) > MinEpsilonH) {
      val alpha1 = epsilonH.div(deltaHRMS)
      alpha.mul(alphaDecay).add(alpha1.mul(1 - alphaDecay))
    } else {
      alpha
    }
    val result = Map(s"h($dimension)" -> h,
      s"pi($dimension)" -> pi,
      s"deltaH($dimension)" -> deltaH,
      s"h*($dimension)" -> hStar,
      s"labels($dimension)" -> actorLabels,
      s"alpha*($dimension)" -> alphaStar)
    result
  }
}

object PolicyActor {
  private val MinEpsilonH = 1e-3

  /**
   * Returns the discrete action agent and the alpha parameters
   *
   * @param conf      the configuration element
   * @param dimension the dimension index
   */
  def fromJson(conf: ACursor)(dimension: Int): Try[(PolicyActor, INDArray)] = {
    for {
      noValues <- conf.get[Int]("noValues").toTry
      range <- rangesFromJson(conf.downField("range"))(1)
      prefRange <- rangesFromJson(conf.downField("prefRange"))(1)
      alpha <- scalarFromJson(conf.downField("alpha"))
      alphaDecay = conf.get[Double]("alphaDecay").getOrElse(1.0)
      epsilon = conf.get[Double]("epsilon").getOrElse(0.1)
    } yield (create(dimension = dimension,
      noValues = noValues,
      actionRange = range,
      prefRange = prefRange,
      alphaDecay = alphaDecay,
      epsilon = epsilon
    ), alpha)
  }

  /**
   * Returns the policy actor
   *
   * @param dimension   the dimension index
   * @param noValues    number of values of action
   * @param actionRange range of action
   * @param prefRange   range of preferences
   * @param epsilon     epsilon parameter
   * @param alphaDecay  alpha decay parameter
   */
  def create(dimension: Int, noValues: Int, actionRange: INDArray, prefRange: INDArray, epsilon: Double, alphaDecay: Double): PolicyActor = {
    val broadPrefRange = prefRange.broadcast(2, noValues)
    val epsilonH = prefRange.getRow(1).sub(prefRange.getRow(0)).muli(epsilon)
    PolicyActor(dimension = dimension,
      noOutputs = noValues,
      alphaDecay = alphaDecay,
      epsilonH = epsilonH,
      denormalize = clipDenormalizeAndCenter(broadPrefRange),
      normalize = clipAndNormalize(broadPrefRange),
      decode = encode(noValues, actionRange),
      encode = decode(noValues, actionRange),
    )
  }
}