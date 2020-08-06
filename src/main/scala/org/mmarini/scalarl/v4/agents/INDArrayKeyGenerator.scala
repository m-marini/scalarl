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
import org.mmarini.scalarl.v4.envs.Configuration._
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j._
import org.nd4j.linalg.ops.transforms.Transforms

import scala.util.{Failure, Success, Try}

/** The factory object for [[INDArrayKeyGenerator]]*/
object INDArrayKeyGenerator {

  val binary: INDArray => ModelKey = x => ModelKey(find(x).map(_.toInt))

  val discrete: INDArray => ModelKey = x => ModelKey(for {
    i <- 0 until x.columns()
  } yield {
    x.getInt(i)
  })

  /**
   * Returns the key generator from json configuration
   *
   * @param cursor the configuration
   * @param noDims the number of dimension
   */
  def fromJson(cursor: ACursor)(noDims: Int): Try[INDArray => ModelKey] = for {
    typ <- cursor.get[String]("type").toTry
    key <- typ match {
      case "Binary" => Success(binary)
      case "Discrete" => Success(discrete)
      case "Tiles" => tilesFromJson(cursor)(noDims)
      case "NormalTiles" => normalTilesFromJson(cursor)(noDims)
      case x => Failure(new IllegalArgumentException(s"Unrecognized key generator type '$x'"))
    }
  } yield key

  /**
   * Returns the tiles key generator from json
   *
   * @param cursor the json tiles configuration
   * @param noDims the number of dimensions
   */
  def tilesFromJson(cursor: ACursor)(noDims: Int): Try[INDArray => ModelKey] = for {
    n <- cursor.get[Array[Int]]("noTiles").toTry
    ranges <- rangesFromJson(cursor.downField("ranges"))(noDims: Int)
    keyEncoder <- Try {
      require(n.length == noDims, s"tiles dimension ${n.length}: must be $noDims")
      val noTiles = create(n.map(_.toDouble))
      tiles(noTiles = noTiles, ranges)

    }
  } yield keyEncoder

  /**
   * Returns the tiles key generator from json
   *
   * @param cursor the json tiles configuration
   * @param noDims the number of dimensions
   */
  def normalTilesFromJson(cursor: ACursor)(noDims: Int): Try[INDArray => ModelKey] = for {
    n <- cursor.get[Array[Int]]("noTiles").toTry
    keyEncoder <- Try {
      require(n.length == noDims, s"tiles dimension ${n.length}: must be $noDims")
      val noTiles = create(n.map(_.toDouble))
      val ranges = create(Array(-1.0, 1.0)).transpose().broadcast(2, noDims)
      tiles(noTiles = noTiles, ranges)
    }
  } yield keyEncoder


  /**
   * Returns the generator
   *
   * @param noTiles the tiles number
   * @param ranges  the continuous ranges
   */
  def tiles(noTiles: INDArray, ranges: INDArray): INDArray => ModelKey = {
    require(noTiles.minNumber().intValue() > 0)
    val min = zeros(noTiles.shape(): _*)
    val toRange = vstack(min, noTiles)
    val tr = transform(ranges, toRange)
    val max = noTiles.sub(1)
    values => {
      val key2 = Transforms.min(tr(values), max)
      val result = for {
        i <- 0 until key2.columns()
      } yield {
        key2.getInt(i)
      }
      ModelKey(result)
    }
  }
}
