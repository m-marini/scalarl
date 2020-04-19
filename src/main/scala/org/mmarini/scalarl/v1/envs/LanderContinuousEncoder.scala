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

package org.mmarini.scalarl.v1.envs

import io.circe.ACursor
import org.mmarini.scalarl.v1._
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j

/**
 * The LanderConf with lander parameters
 *
 * @param zMax    m the maximum admitted height m
 * @param hRange  the maximum admitted horizontal range
 * @param vhRange the horizontal speed scale
 * @param vzRange the vertical speed scale
 */
class LanderContinuousEncoder(zMax: Double,
                              hRange: Double,
                              vhRange: Double,
                              vzRange: Double) extends LanderEncoder {

  /** Returns the number of signals */
  override val noSignals: Int = 6
  private val signalsFromPos = {
    val result = Nd4j.zeros(3L, 3L)
    result.put(0, 0, 1 / hRange)
    result.put(1, 1, 1 / hRange)
    result.put(2, 2, 1 / zMax)
    result
  }
  private val signalsFromSpeed = {
    val result = Nd4j.zeros(3L, 3L)
    result.put(0, 0, 1 / vhRange)
    result.put(1, 1, 1 / vhRange)
    result.put(2, 2, 1 / vzRange)
    result
  }

  /**
   * Returns the input signals
   *
   * @param status the status
   */
  override def signals(status: LanderStatus): QValues = {
    val posSignals: INDArray = status.pos.mmul(signalsFromPos)
    val speedSignals: INDArray = status.speed.mmul(signalsFromSpeed)
    val signals = Nd4j.hstack(
      posSignals,
      speedSignals)
    signals
  }
}

/** Factory for [[LanderContinuousEncoder]] instances */
object LanderContinuousEncoder {
  /**
   *
   * @param conf the json configuration
   */
  def apply(conf: ACursor): LanderContinuousEncoder = new LanderContinuousEncoder(
    hRange = conf.get[Double]("hRange").right.get,
    zMax = conf.get[Double]("zMax").right.get,
    vhRange = conf.get[Double]("vhRange").right.get,
    vzRange = conf.get[Double]("vzRange").right.get)
}
