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

import io.circe.ACursor
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j._

import scala.util.Try

/**
 * Factory for reward functions
 */
object LanderRewards {
  /**
   * Returns the reward function
   *
   * @param conf the json configuration
   */
  def fromJson(conf: ACursor): Try[INDArray => INDArray] =
    paramsFromJson(conf).map(apply)

  /**
   * Returns the parameters of lander reward function in the order:
   * (base, direction, distance, height, hSpeed, vSpeed)
   *
   * @param conf the json configuration
   */
  def paramsFromJson(conf: ACursor): Try[INDArray] = for {
    base <- conf.get[Double]("base").toTry
  } yield {
    val direction = conf.get[Double]("direction").getOrElse(0.0)
    val distance = conf.get[Double]("distance").getOrElse(0.0)
    val height = conf.get[Double]("height").getOrElse(0.0)
    val hSpeed = conf.get[Double]("hSpeed").getOrElse(0.0)
    val vSpeed = conf.get[Double]("vSpeed").getOrElse(0.0)
    val result = create(Array(base, direction, distance, height, hSpeed, vSpeed))
    result
  }

  /**
   * Returns the reward function
   *
   * @param params the parameters
   */
  def apply(params: INDArray): INDArray => INDArray =
    (s: INDArray) =>
      params.mmul(s.transpose())
}
