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

import io.circe.ACursor
import org.mmarini.scalarl.v6.Configuration._
import org.mmarini.scalarl.v6.Utils._
import org.nd4j.linalg.api.ndarray.INDArray

import scala.util.{Failure, Success, Try}

/** The factory object for [[INDArrayKeyGenerator]] */
object INDArrayKeyGenerator {

  val binary: INDArray => ModelKey = x => ModelKey(find(x).map(_.toInt))

  /**
   * Returns the key generator from json configuration.
   * Allowed type:
   * Binary: Vector of binary values e.g. [0 1 0 0 1 ] => [1 4]
   * Discrete: Vector of discrete values (ranges for each dimension, noValues for each dimension) e.g. [0 1 2 3] -> [0 1 2 3]
   * Tiles: Vector of real values mapped with tiles (noTiles for each dimension, value ranges for each dimension )
   *
   * @param cursor the configuration
   * @param noDims the number of dimension
   */
  def fromJson(cursor: ACursor)(noDims: Int): Try[INDArray => ModelKey] = for {
    typ <- cursor.get[String]("type").toTry
    key <- typ match {
      case "Binary" => Success(binary)
      case "Discrete" => discreteFromJson(cursor)(noDims)
      case x => Failure(new IllegalArgumentException(s"Unrecognized key generator type '$x'"))
    }
  } yield key

  /**
   * Returns the discrete key generation function
   *
   * @param cursor the configuration
   * @param noDims the number of dimensions
   */
  def discreteFromJson(cursor: ACursor)(noDims: Int): Try[INDArray => ModelKey] = for {
    fromRanges <- rangesFromJson(cursor.downField("ranges"))(noDims)
    noValues <- posIntVectorFromJson(cursor.downField("noValues"))(noDims)
  } yield {
    discrete(fromRanges, noValues)
  }

  /**
   * Returns the function that creates a ModelKey from an continuous input vector
   *
   * @param fromRanges input ranges
   * @param noValues   number of values for each dimensions
   */
  def discrete(fromRanges: INDArray, noValues: INDArray): INDArray => ModelKey = {
    val noDims = fromRanges.size(1).toInt
    val (encode, _) = Encoder.discrete(ranges = fromRanges, noTiles = noValues)

    (x: INDArray) => {
      val y = encode(x)
      ModelKey(for {
        i <- 0 until noDims
      } yield {
        y.getInt(i)
      })
    }
  }
}
