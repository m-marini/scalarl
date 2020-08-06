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

package org.mmarini.scalarl.v4.envs

import io.circe.ACursor
import org.mmarini.scalarl.v4.Utils
import org.nd4j.linalg.api.ndarray.INDArray

/**
 * The LanderConf with lander parameters
 *
 */
class LanderTilesEncoder(hash: Option[Int], normalize: INDArray => INDArray) extends LanderEncoder {
  /** Returns the number of signals */
  private val tilesCoder: Tiles = hash.map(h => Tiles.withHash(h, 1, 1, 1, 1, 1, 1)).getOrElse(Tiles(1, 1, 1, 1, 1, 1))
  override val noSignals: Int = tilesCoder.noFeatures.toInt
  private val Size = 6

  /**
   * Returns the input signals
   *
   * @param in the input signals
   */
  override def signals(in: INDArray): INDArray = {
    val s = normalize(in)
    val features = tilesCoder.features(s)
    val signals = Utils.features(features, tilesCoder.noFeatures)
    signals
  }
}

/** Factory for [[LanderTilesEncoder]] instances */
object LanderTilesEncoder {
  /**
   * Returns the tile encoder
   *
   * @param conf   the json configuration
   * @param ranges the input ranges
   * @return
   */
  def fromJson(conf: ACursor, ranges: INDArray) =
    new LanderTilesEncoder(hash = conf.get[Int]("hash").toOption, Utils.normalize01(ranges))
}
