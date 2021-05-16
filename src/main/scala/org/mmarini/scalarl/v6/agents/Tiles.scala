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

package org.mmarini.scalarl.v6.agents

import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.ops.transforms.Transforms

import scala.math._

/**
 * Generates tiles code for multi dimension spaces
 *
 * @param offsets the tilings offsets
 * @param limits  the limits
 * @param strides the stride for each dimension
 * @param noTiles the number of tiles
 * @param hash    the number of hash
 */
class Tiles(val offsets: INDArray, limits: INDArray, strides: INDArray, noTiles: Long, hash: Option[Int]) {
  /** Returns the number of features */
  val noFeatures: Long = hash.map(h => min(noTiles * tilings, h)).getOrElse(noTiles * tilings)
  private val Seed = (1L << 17) - 1L // Mersenne prime

  /**
   * Converts the point into a tile features
   * The points coordinate should have range 0 ... limits(i)
   *
   * @param point the point
   */
  def features(point: INDArray): Seq[Long] = {
    val clip = Transforms.max(point, 0)
    val f = for {
      i <- 0L until tilings
    } yield {
      val p = clip.add(offsets.getRow(i))
      val tiles = Transforms.min(Transforms.max(Transforms.floor(p), 0), limits)
      val idx = tiles.mul(strides)
      val n = idx.sumNumber().longValue() + i * noTiles
      hash.map(h => (n * Seed) % h).getOrElse(n)
    }
    hash.map(_ => f.distinct).getOrElse(f)
  }

  /** Returns the number of tilings */
  def tilings: Long = offsets.size(0)
}

/** Factory for [[Tiles]] */
object Tiles {
  private val Primes = Seq(1, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 43, 47, 53, 59).map(_.toLong)

  /**
   * Returns coarse tile code
   *
   * @param hash  the hash size
   * @param sizes sizes per dimension
   */
  def withHash(hash: Int, sizes: Long*): Tiles = apply(Some(hash), sizes: _*)

  /**
   * Returns coarse tile code
   *
   * @param sizes sizes per dimension
   */
  def apply(sizes: Long*): Tiles = apply(None, sizes: _*)

  /**
   * Returns coarse tile code
   * The space coordinates are limited to 0 ... size(1)
   *
   * @param hash  The hash size
   * @param sizes sizes per dimension
   */
  def apply(hash: Option[Int], sizes: Long*): Tiles = {
    require(sizes.nonEmpty)
    sizes.foreach(s => require(s > 0))

    // Number of dimensions
    val k = sizes.length.toLong

    // Number of tile sets
    val ne = ceil(log(4 * k) / log(2)).toLong
    val n = pow(2, ne).toLong

    // Compute the offsets vectors
    val x = for {
      i <- 0 until n.toInt
    } yield {
      val ary = for {s <- Primes.take(k.toInt)} yield {
        ((s * i) % n).toDouble / n
      }
      ary.toArray
    }
    // An offset row vector for each tile set
    val offsets = Nd4j.create(x.toArray)

    // Computes the total number of tiles for each tile set
    val sizeV = Nd4j.create(sizes.map(_.toDouble).toArray)
    val vSize = sizeV.add(1)
    val noTiles = vSize.prodNumber().longValue()

    // Compute the stride for each dimension
    // 1, s(1), s(1) * s(2), ..., s(n-1) * s(n)
    val strides = Nd4j.ones(vSize.shape(): _*)
    for {
      i <- 1L until vSize.length()
    } {
      strides.putScalar(i, strides.getInt(i.toInt - 1) * vSize.getInt(i.toInt))
    }
    new Tiles(offsets, sizeV, strides, noTiles, hash)
  }
}
