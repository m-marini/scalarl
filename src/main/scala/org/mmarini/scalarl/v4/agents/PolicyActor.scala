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
 * Actor critic agent
 *
 * @param dimension   the dimension index
 * @param noOutputs   the number of value for action
 * @param denormalize the output denormalizer function to preference
 * @param normalize   the output normalizer function from preference
 * @param transform   the action transformation function
 * @param inverse     the action inverse transformation function
 * @param alpha       the alpha parameter
 */
case class PolicyActor(dimension: Int,
                       noOutputs: Int,
                       denormalize: INDArray => INDArray,
                       normalize: INDArray => INDArray,
                       transform: INDArray => INDArray,
                       inverse: INDArray => INDArray,
                       alpha: INDArray) extends Actor with LazyLogging {

  /**
   * Returns the action choosen by the actor
   *
   * @param outputs the network outputs
   * @param random  the random generator
   */
  override def chooseAction(outputs: Array[INDArray], random: Random): INDArray = {
    val pi = softmax(preferences(outputs))
    val action = transform(ones(1).muli(randomInt(pi)(random)))
    action
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
                             delta: INDArray): Map[String, Any] = {
    val h = preferences(outputs)
    val pi = softmax(h)
    val action = inverse(actions.getScalar(dimension.toLong)).getInt(0)
    val z = features(Seq(action), h.length()).subi(pi)
    val deltaH = z.mul(delta).muli(alpha)
    val hStar = h.add(deltaH)
    val actorLabels = normalize(hStar)
    val result = Map(s"h($dimension)" -> h,
      s"pi($dimension)" -> pi,
      s"deltaH($dimension)" -> deltaH,
      s"h*($dimension)" -> hStar,
      s"labels($dimension)" -> actorLabels)
    result
  }

  /**
   * Returns the preferences
   *
   * @param outputs the outputs
   */
  def preferences(outputs: Array[INDArray]): INDArray =
    preferences(outputs(dimension + 1))

  /**
   * Returns the preferences
   *
   * @param outputs the outputs
   */
  def preferences(outputs: INDArray): INDArray = {
    val pr = denormalize(outputs)
    val pr1 = pr.sub(pr.mean())
    pr1
  }
}

object PolicyActor {

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
                              modelPath: Option[String]): Try[PolicyActor] = {
    for {
      noValues <- conf.get[Int]("noValues").toTry
      range <- rangesFromJson(conf.downField("range"))(1)
      prefRange <- rangesFromJson(conf.downField("prefRange"))(1)
      alpha <- scalarFromJson(conf.downField("alpha"))
    } yield {
      val broadPrefRange = prefRange.broadcast(2, noValues)
      val transfom = denormalize(broadPrefRange)
      val grad = normalize(broadPrefRange)
      val fromRange = create(Array[Double](0, noValues - 1)).transposei()
      PolicyActor(dimension = dimension,
        noOutputs = noValues,
        denormalize = transfom,
        normalize = grad,
        alpha = alpha,
        transform = transform(fromRange, range),
        inverse = transform(range, fromRange)
      )
    }
  }
}