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

package org.mmarini.scalarl.v3.envs

import io.circe.ACursor
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j._

/**
 * The LanderConf with lander parameters
 *
 * @param statusScale the status scale
 */
class LanderContinuousEncoder(statusScale: INDArray) extends LanderEncoder {

  /**
   * Returns the input signals
   *
   * @param status the status
   */
  override def signals(status: LanderStatus): INDArray =
    hstack(status.pos, status.speed).muli(statusScale)

  /** Returns the number of signals */
  override val noSignals: Int = 6
}

/** Factory for [[LanderContinuousEncoder]] instances */
object LanderContinuousEncoder {
  /**
   *
   * @param conf the json configuration
   */
  def fromJson(conf: ACursor): LanderContinuousEncoder = {
    val hRange = conf.get[Double]("hRange").toTry.get
    val zMax = conf.get[Double]("zMax").toTry.get
    val vhRange = conf.get[Double]("vhRange").toTry.get
    val vzRange = conf.get[Double]("vzRange").toTry.get
    new LanderContinuousEncoder(create(Array(
      1 / hRange, 1 / hRange, 1 / zMax,
      1 / vhRange, 1 / vhRange, 1 / vzRange
    )))
  }
}
