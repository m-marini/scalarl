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
import org.mmarini.scalarl.v1.Utils
import org.mmarini.scalarl.v1.envs.LanderTilesEncoder.{HPrecision, VHPrecision, VZPrecision, ZPrecision}
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j

/**
 * The LanderConf with lander parameters
 *
 */
class LanderTilesEncoder(hash: Option[Int]) extends LanderEncoder {
  private val Size = 6
  private val tilesCoder: Tiles = hash.map(h => Tiles.withHash(h, 1, 1, 1, 1, 1, 1)).getOrElse(Tiles(1, 1, 1, 1, 1, 1))
  //  private val vhOffset = VHPrecision * tilesCoder.tilings / 2
  private val statusOffset: INDArray = Nd4j.create(Array(-HPrecision, -HPrecision, 0, -VHPrecision, -VHPrecision, -VZPrecision)).
    mul(tilesCoder.tilings / 2)
  private val statusScale: INDArray = Nd4j.ones(Size).div(tilesCoder.tilings).
    div(Nd4j.create(Array(HPrecision, HPrecision, ZPrecision, VHPrecision, VHPrecision, VZPrecision)))

  /** Returns the number of signals */
  override val noSignals: Int = tilesCoder.noFeatures.toInt

  /**
   * Returns the input signals
   *
   * @param status the status
   */
  override def signals(status: LanderStatus): INDArray = {
    val statusVect = Nd4j.hstack(status.pos, status.speed)
    val s0 = statusVect.sub(statusOffset)
    val s = s0.mul(statusScale)
    val features = tilesCoder.features(s)
    val signals = Utils.features(features, tilesCoder.noFeatures)
    signals

  }
}

/** Factory for [[LanderTilesEncoder]] instances */
object LanderTilesEncoder {
  val HPrecision = 4.0
  val ZPrecision = 1.0
  val VHPrecision = 0.4
  val VZPrecision = 0.5

  /**
   *
   * @param conf the json configuration
   */
  def apply(conf: ACursor): LanderTilesEncoder = new LanderTilesEncoder(hash = conf.get[Int]("hash").toOption)
}
