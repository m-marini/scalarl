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
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j

/**
 * The custom status encoder
 *
 * @param zMax          the maximum admitted height m
 * @param z1            the sensor height thresold
 * @param landingRadius the radius of landing area m
 * @param landingVH     the maximum horizontal landing speed m/s
 * @param landingVZ     the maximum vertical landing speed m/s
 * @param hRange        the maximum admitted horizontal range
 * @param vhRange       the horizontal speed scale
 * @param vzRange       the vertical speed scale
 */
class LanderCustomEncoder(zMax: Double,
                          z1: Double,
                          landingRadius: Double,
                          landingVH: Double,
                          landingVZ: Double,
                          hRange: Double,
                          vhRange: Double,
                          vzRange: Double) extends LanderEncoder {

  /** Returns the number of signals */
  override val noSignals: Int = 28
  private val squaredRadius = landingRadius * landingRadius
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
  override def signals(status: LanderStatus): INDArray = {
    val pos = status.pos
    val speed = status.speed
    val x = pos.getDouble(0L)
    val y = pos.getDouble(1L)
    val z = pos.getDouble(2L)

    val vx = speed.getDouble(0L)
    val vy = speed.getDouble(1L)
    val vz = speed.getDouble(2L)

    val posSignals: INDArray = pos.mmul(signalsFromPos)
    val speedSignals: INDArray = speed.mmul(signalsFromSpeed)

    // Computes the position feature index
    val radius2 = x * x + y * y
    val xIdx = if (x > 0) 1 else 0
    val yIdx = if (y > 0) 1 else 0
    val zIdx = if (z >= z1) 1 else 0
    val posIdx = if (radius2 >= squaredRadius) {
      yIdx + 2 * xIdx + 4 * zIdx
    } else {
      8 + zIdx
    }
    val posFeatures = Nd4j.zeros(10)
    posFeatures.put(0, posIdx, 1)

    // Computes the horizontal speed feature index
    val vh = vx * vx + vy * vy
    val vxIdx = if (vx > 0) 1 else 0
    val vyIdx = if (vy > 0) 1 else 0
    val v02 = v0 * v0
    val vhhIdx = if (vh >= v02) 1 else 0
    val vhIdx = vyIdx + 2 * vxIdx + 4 * vhhIdx
    val vhFeatures = Nd4j.zeros(8)
    vhFeatures.put(0, vhIdx, 1)

    // Computes the vertical speed features
    val vzFeatures = Nd4j.zeros(4)
    if (vz >= 0) {
      vzFeatures.put(0, 0, 1)
    } else if (vz > v1) {
      vzFeatures.put(0, 1, 1)
    } else if (vz > v2) {
      vzFeatures.put(0, 2, 1)
    } else {
      vzFeatures.put(0, 3, 1)
    }

    val signals = Nd4j.hstack(
      posSignals,
      speedSignals,
      posFeatures,
      vhFeatures,
      vzFeatures)
    signals
  }

  /** Returns the feature safe horizontal speed */
  private def v0 = landingVH / 2

  /** Returns the feature safe vertical speed */
  private def v1 = -landingVZ / 2

  /** Returns the feature max vertical speed */
  private def v2 = -landingVZ
}

/** Factory for [[LanderCustomEncoder]] instances */
object LanderCustomEncoder {
  /**
   *
   * @param conf the json configuration
   */
  def fromJson(conf: ACursor): LanderCustomEncoder = {
    new LanderCustomEncoder(z1 = conf.get[Double]("z1").toTry.get,
      hRange = conf.get[Double]("hRange").toTry.get,
      zMax = conf.get[Double]("zMax").toTry.get,
      landingRadius = conf.get[Double]("landingRadius").toTry.get,
      landingVH = conf.get[Double]("landingVH").toTry.get,
      landingVZ = conf.get[Double]("landingVZ").toTry.get,
      vhRange = conf.get[Double]("vhRange").toTry.get,
      vzRange = conf.get[Double]("vzRange").toTry.get
    )
  }
}
