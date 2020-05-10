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

package org.mmarini.scalarl.v3.agents

import io.circe.ACursor
import org.mmarini.scalarl.v3.Utils
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j._
import org.nd4j.linalg.ops.transforms.Transforms
import org.nd4j.linalg.ops.transforms.Transforms._

/**
 *
 * @param offset the offset
 * @param range  the range
 * @param dims   the dimensions
 */
class INDArrayKeyGenerator(offset: INDArray,
                           range: INDArray,
                           dims: INDArray) {

  private val scale1 = dims.div(range)

  /**
   * Returns the key of a vector
   *
   * @param values the key vector
   */
  def build(values: INDArray): INDArray = {
    val key1 = values.sub(offset).muli(scale1)
    val key2 = round(Transforms.min(Transforms.max(zeros(dims.shape(): _*), key1), dims))
    key2
  }
}

/** The factory object for [[INDArrayKeyGenerator]] */
object INDArrayKeyGenerator {
  /**
   * Returns the key generator from json configuration
   *
   * @param cursor the configuration
   */
  def fromJson(cursor: ACursor): INDArray => INDArray = cursor.get[String]("type").toTry.get match {
    case "Identity" => x => x
    case "Sparse" => x => create(Utils.find(x).map(_.toDouble).toArray)
    case x => throw new IllegalArgumentException(s"Unrecognized key generator type '$x'")
  }

  /**
   * Returns the generator
   *
   * @param min     the state minimum values
   * @param max     the state maximum values
   * @param noTiles the state tiles number
   */
  def apply(min: INDArray,
            max: INDArray,
            noTiles: INDArray): INDArrayKeyGenerator = {
    require(min.equalShapes(max))
    require(min.equalShapes(noTiles))
    require(noTiles.minNumber().intValue() > 0)

    new INDArrayKeyGenerator(min, max.sub(min), noTiles)
  }
}
