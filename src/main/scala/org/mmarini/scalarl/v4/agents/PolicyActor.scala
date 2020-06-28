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
import org.mmarini.scalarl.v4.Utils._
import org.mmarini.scalarl.v4.agents.PolicyActor._
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.api.rng.Random
import org.nd4j.linalg.factory.Nd4j._
import org.nd4j.linalg.ops.transforms.Transforms._

/**
 * Actor critic agent
 *
 * @param dimension the dimension index
 * @param noOutputs the number of value for action
 * @param alpha     the alpha parameter
 */
case class PolicyActor(dimension: Int,
                       noOutputs: Int,
                       alpha: INDArray) extends Actor with LazyLogging {

  /**
   * Returns the action choosen by the actor
   *
   * @param outputs the network outputs
   * @param random  the random generator
   */
  override def chooseAction(outputs: Array[INDArray], random: Random): INDArray = {
    val pi = softmax(preferences(outputs))
    val action = ones(1).muli(randomInt(pi)(random))
    action
  }

  /**
   * Returns the actor labels
   *
   * @param outputs the outputs
   * @param actions the actions
   * @param delta   the td error
   * @param random  the random generator
   */
  override def computeLabels(outputs: Array[INDArray],
                             actions: INDArray,
                             delta: INDArray,
                             random: Random): INDArray = {
    val prefs = preferences(outputs)
    val actorLabels = computeActorLabel(prefs, actions.getInt(dimension), alpha, delta)
    actorLabels
  }

  /**
   * Returns the preferences
   *
   * @param outputs the outputs
   */
  def preferences(outputs: Array[INDArray]): INDArray =
    normalize(outputs(dimension + 1))
}

object PolicyActor {
  val PreferenceRange = 7

  /**
   * Returns the actor target label
   *
   * @param prefs  the preferences
   * @param action the action
   * @param alpha  the alpha parameter
   * @param delta  the TD Error
   */
  def computeActorLabel(prefs: INDArray, action: Int, alpha: INDArray, delta: INDArray): INDArray = {
    val pi = softmax(prefs)
    val expTot = exp(prefs).sum()
    // deltaH = (A_i(a) / expTot - pi) alpha delta
    val A = features(Seq(action), prefs.length()).divi(expTot)
    val deltaPref = A.sub(pi).muli(alpha).muli(delta)
    val actorLabel = normalize(prefs.add(deltaPref))
    actorLabel
  }

  /**
   * Returns the normalized preferences
   *
   * @param data the preferences
   */
  def normalize(data: INDArray): INDArray =
    scaleClip(data.sub(mean(data)), PreferenceRange)
}