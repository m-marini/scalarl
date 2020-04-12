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

package org.mmarini.scalarl.ts.envs

import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.indexing.NDArrayIndex
import org.nd4j.linalg.ops.transforms.Transforms

import scala.math._

class Tiles(val offset: INDArray, val n: Int) {

  def indices(point: INDArray): Seq[INDArray] = {
    val tiles = for {
      tile <- 0 until n
    } yield {
      val x1 = Transforms.floor(offset.mul(tile).addi(point))
      x1
    }
    tiles
  }
}

object Tiles {
  val Primes = Nd4j.create(Array(1, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 43, 47, 53, 59).map(_.toDouble))

  def main(args: Array[String]): Unit = {
    val t = Tiles(2)
    println(t.offset)
    println(t.n)
    println(t.indices(Nd4j.create(Array(0.0, 0.0))))
    println(t.indices(Nd4j.create(Array(0.2, 0.0))))
    println(t.indices(Nd4j.create(Array(0.3, 0.0))))
    println(t.indices(Nd4j.create(Array(0.6, 0.0))))
    println(t.indices(Nd4j.create(Array(0.8, 0.0))))
    println(t.indices(Nd4j.create(Array(1.0, 0.0))))
  }


  /**
   * Returns corse tile code
   *
   * @param k number of dimensions
   */
  def apply(k: Int): Tiles = {
    val ne = ceil(log(4 * k) / log(2)).toInt
    val n = pow(2, ne).toInt
    val strides = Primes.get(NDArrayIndex.interval(0, k))
    new Tiles(strides, n)
  }
}