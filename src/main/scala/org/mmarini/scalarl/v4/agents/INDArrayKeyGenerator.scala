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

package org.mmarini.scalarl.v4.agents

import io.circe.ACursor
import org.mmarini.scalarl.v4.Utils._
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j._
import org.nd4j.linalg.ops.transforms.Transforms
import org.nd4j.linalg.ops.transforms.Transforms._

/** The factory object for [[INDArrayKeyGenerator]] */
object INDArrayKeyGenerator {

  val binary: INDArray => Seq[Int] = x => find(x).map(_.toInt).toArray

  val discrete: INDArray => Seq[Int] = x => (for {
    i <- 0 until x.columns()
  } yield {
    x.getInt(i)
  }).toArray

  /**
   * Returns the key generator from json configuration
   *
   * @param cursor the configuration
   * @param noDims the number of dimension
   */
  def fromJson(cursor: ACursor)(noDims: Int): INDArray => Seq[Int] = cursor.get[String]("type").toTry.get match {
    case "Binary" => binary
    case "Discrete" => discrete
    case "Tiles" => tilesFromJson(cursor)(noDims)
    case x => throw new IllegalArgumentException(s"Unrecognized key generator type '$x'")
  }

  /**
   * Returns the tiles key generator from json
   *
   * @param cursor the json tiles configuration
   * @param noDims the number of dimensions
   */
  def tilesFromJson(cursor: ACursor)(noDims: Int): INDArray => Seq[Int] = {
    val offset = create(cursor.get[Array[Double]]("offset").toTry.get)
    require(offset.shape() sameElements Array(1L, noDims))
    val max = create(cursor.get[Array[Double]]("max").toTry.get)
    require(max.shape() sameElements Array(1L, noDims))
    val noTiles = create(cursor.get[Array[Int]]("tiles").toTry.get.map(_.toDouble))
    require(noTiles.shape() sameElements Array(1L, noDims))
    tiles(
      min = offset,
      max = max,
      noTiles = noTiles)
  }

  /**
   * Returns the generator
   *
   * @param min     the state minimum values
   * @param max     the state maximum values
   * @param noTiles the state tiles number
   */
  def tiles(min: INDArray,
            max: INDArray,
            noTiles: INDArray): INDArray => Seq[Int] = {
    require(min.equalShapes(max))
    require(min.equalShapes(noTiles))
    require(noTiles.minNumber().intValue() > 0)

    val scale1 = noTiles.div(max.sub(min))
    values => {
      val key1 = values.sub(min).muli(scale1)
      val key2 = round(Transforms.min(Transforms.max(zeros(noTiles.shape(): _*), key1), noTiles))
      val result = for {
        i <- 0 until key2.columns()
      } yield {
        key2.getInt(i)
      }
      result.toArray
    }
  }
}
