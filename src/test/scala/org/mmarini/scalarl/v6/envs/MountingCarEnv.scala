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
import org.mmarini.scalarl.v6.envs.MountingCarEnv._
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.api.rng.Random
import org.nd4j.linalg.factory.Nd4j._
import org.nd4j.linalg.ops.transforms.Transforms
import org.nd4j.linalg.ops.transforms.Transforms._

/**
 * The mounting car state
 *
 * @param x position
 * @param v speed
 * @param t time
 */
case class MountingCarEnv(x: INDArray,
                          v: INDArray,
                          t: INDArray) extends Env {

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
    if (x.getDouble(0L) >= XRight) {
      // Restart from random position, speed=0
      (initial(random, t.add(1)), zeros(1))
    } else {
      val clipAction = clip(action, MinActionValue, MaxActionValue)
      // v' = v + 1e-3 action - 2.5e-3 cos(3 x)
      val v1 = v.add(clipAction.mul(1e-3)).subi(cos(x.mul(3)).muli(2.5e-3))
      val v1Clip = clip(v1, VMin, VMax)
      // x' = x + v'
      val x1 = x.add(v1Clip)
      val x1Clip = clip(x1, XLeft, XRight)
      val v2 = if (x1Clip.getDouble(0L) > XLeft) v1Clip else zeros(1)
      val reward = if (x1Clip.getDouble(0L) >= XRight) ones(1) else ones(1).negi()
      (copy(x = x1Clip, v = v2, t = t.add(1)), reward)
    }
  }

  /** Returns the number of signals */
  override def signalsSize: Int = 2

  /** Returns the [[Observation]] for the environment */
  override def observation: Observation =
    INDArrayObservation(
      signals = hstack(x, v),
      time = t)

  /** Returns the action configuration */
  override def actionDimensions: Int = 1
}

object MountingCarEnv {
  val XLeft: Double = -1.2
  val XRight: Double = 0.5
  val VMin: Double = -0.07
  val VMax = 0.07
  val X0Min: Double = -0.6
  val X0Max: Double = -0.4
  val MinActionValue: Double = -1.0
  val MaxActionValue: Double = 1.0

  /**
   *
   * @param random the random generator
   */
  def initial(random: Random): MountingCarEnv = initial(random, zeros(1))

  /**
   *
   * @param random the random generator
   */
  def initial(random: Random, t: INDArray): MountingCarEnv = {
    val x0 = create(Array(random.nextDouble() * (XRight - XLeft) + XLeft))
    MountingCarEnv(x = x0, v = zeros(1), t = t)
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
}