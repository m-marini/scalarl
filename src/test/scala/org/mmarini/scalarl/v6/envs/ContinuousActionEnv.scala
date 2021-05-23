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

package org.mmarini.scalarl.v6.envs

import org.mmarini.scalarl.v6._
import org.mmarini.scalarl.v6.envs.ContinuousActionEnv.{MaxActionValue, MaxStateValue, MinActionValue, MinStateValue}
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.api.rng.Random
import org.nd4j.linalg.ops.transforms.Transforms

case class ContinuousActionEnv(x: INDArray, t: INDArray) extends Env {

  /**
   * Computes the next status of environment executing an action.
   *
   * @param action the executing action
   * @param random the random generator
   * @return a n-tuple with:
   *         - the environment in the next status,
   *         - the reward for the action,
   */
  override def change(action: INDArray, random: Random): (Env, INDArray) = {
    val clipAction = clip(action, MinActionValue, MaxActionValue)
    val reward = Transforms.pow(clipAction.sub(x), 2).negi()
    val x1 = random.nextDouble(Array(1, 1)).muli(MaxStateValue - MinStateValue).addi(MinStateValue)
    (copy(t = t.add(1),
      x = x1),
      reward)
  }

  /**
   * Returns the clip values
   *
   * @param x    the values
   * @param xMin minimum value
   * @param xMax maximum values
   * @param copy true if return value is a new copy
   */
  def clip(x: INDArray, xMin: Double, xMax: Double, copy: Boolean = true): INDArray = Transforms.min(Transforms.max(x, xMin, copy), xMax, copy)

  /** Returns the number of signals */
  override def signalsSize: Int = 1

  /** Returns the [[Observation]] for the environment */
  override def observation: Observation = INDArrayObservation(
    signals = x,
    time = t)

  /** Returns the action space dimension */
  override def actionDimensions: Int = 1
}

object ContinuousActionEnv {
  val MinStateValue = -2.0
  val MaxStateValue = 2.0
  val MinActionValue = -2.0
  val MaxActionValue = 2.0
}
