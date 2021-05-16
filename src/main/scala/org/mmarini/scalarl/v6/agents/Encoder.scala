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
import org.mmarini.scalarl.v6.Utils
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j.{ones, vstack, zeros}
import org.nd4j.linalg.ops.transforms.Transforms

import scala.util.{Failure, Success, Try}

/** Factory for [[Encoder]] instances */
object Encoder {

  /**
   * Returns the encoder and the number of network input signals
   *
   * @param conf             the json configuration
   * @param signalDimensions the number of signal dimensions
   */
  def fromJson(conf: ACursor)(signalDimensions: Int): Try[(INDArray => INDArray, Int)] = {
    val result = for {
      typ <- conf.get[String]("type").toTry
      encoder <-
        typ match {
          case "Tiles" => tilesFromJson(conf, signalDimensions)
          case "Discrete" => discreteFromJson(conf, signalDimensions)
          case "Features" => featuresFromJson(conf, signalDimensions)
          case "Continuous" =>
            rangesFromJson(conf.downField("ranges"))(signalDimensions).map(continuous)
          case typ => Failure(new IllegalArgumentException(s"Unrecognized coder type '$typ'"))
        }
    }
    yield encoder
    result
  }

  /**
   * Returns the function that encode the inputs to continuous normalized values
   *
   * @param ranges the ranges of each dimension
   */
  def continuous(ranges: INDArray): (INDArray => INDArray, Int) =
    (Utils.clipAndNormalize(ranges = ranges), ranges.size(1).toInt)

  /**
   * Returns the tile encoder
   *
   * @param conf   the json configuration
   * @param noDims number of dimension
   */
  def tilesFromJson(conf: ACursor, noDims: Int): Try[(INDArray => INDArray, Int)] = {
    for {
      ranges <- rangesFromJson(conf.downField("ranges"))(noDims)
      noTilesCurs = conf.downField("noTiles")
      sizes <- if (noTilesCurs.failed) {
        Success(ones(noDims))
      } else {
        posIntVectorFromJson(noTilesCurs)(noDims)
      }
    } yield {
      val hash = conf.get[Int]("hash").toOption
      tiles(ranges, sizes, hash)
    }
  }

  /**
   * Returns the tiles encoder and the number of features
   *
   * @param ranges the ranges of each dimension
   * @param sizes  the number of tiles for each dimension
   * @param hash   the hash value
   */
  def tiles(ranges: INDArray, sizes: INDArray, hash: Option[Int]): (INDArray => INDArray, Int) = {
    val noDims = ranges.size(1).toInt
    val sizeArray = for {i <- 0 until noDims} yield {
      sizes.getInt(i).toLong
    }
    val tilesCoder: Tiles = Tiles(hash, sizeArray: _ *)

    val toRange = vstack(zeros(noDims), sizes)
    val normalize = Utils.clipAndTransform(ranges, toRange)
    val encoder = (in: INDArray) => {
      val s = normalize(in)
      val features = tilesCoder.features(s)
      val signals = Utils.features(features, tilesCoder.noFeatures)
      signals
    }
    (encoder, tilesCoder.noFeatures.toInt)
  }

  /**
   * Returns the tile encoder
   *
   * @param conf   the json configuration
   * @param noDims number of dimension
   */
  def discreteFromJson(conf: ACursor, noDims: Int): Try[(INDArray => INDArray, Int)] = {
    for {
      ranges <- rangesFromJson(conf.downField("ranges"))(noDims)
      noTilesCurs = conf.downField("noTiles")
      sizes <- if (noTilesCurs.failed) {
        Success(ones(noDims))
      } else {
        posIntVectorFromJson(noTilesCurs)(noDims)
      }
    } yield {
      discrete(ranges, sizes)
    }
  }

  /**
   * Returns the discrete encoder and the number of features
   *
   * @param ranges  the ranges of each dimension
   * @param noTiles the number of tiles for each dimension
   */
  def discrete(ranges: INDArray, noTiles: INDArray): (INDArray => INDArray, Int) = {
    val noDims = ranges.size(1).toInt
    val toRanges = vstack(zeros(noDims), noTiles)
    val transform = Utils.clipAndTransform(ranges, toRanges)
    val max = noTiles.sub(1)
    val result = (x: INDArray) => {
      val y = transform(x)
      val z = Transforms.min(Transforms.floor(y), max)
      z
    }
    (result, noDims)
  }

  /**
   * Returns the tile encoder
   *
   * @param conf   the json configuration
   * @param noDims number of dimension
   */
  def featuresFromJson(conf: ACursor, noDims: Int): Try[(INDArray => INDArray, Int)] = {
    for {
      ranges <- rangesFromJson(conf.downField("ranges"))(noDims)
      noTilesCurs = conf.downField("noTiles")
      sizes <- if (noTilesCurs.failed) {
        Success(ones(noDims))
      } else {
        posIntVectorFromJson(noTilesCurs)(noDims)
      }
    } yield {
      features(ranges, sizes)
    }
  }

  /**
   * Returns the single tile set encoder and the number of features
   *
   * @param ranges the ranges of each dimension
   * @param sizes  the number of tiles for each dimension
   */
  def features(ranges: INDArray, sizes: INDArray): (INDArray => INDArray, Int) = {
    val (fDiscrete, _) = discrete(ranges, sizes)
    val noDims = ranges.size(1).toInt
    val stride = ones(noDims)
    for {i <- 1 until noDims} {
      stride.putScalar(i, sizes.getInt(i - 1) * stride.getInt(i - 1))
    }
    val noFeatures = sizes.prodNumber().intValue()
    val result = (x: INDArray) => {
      val z = fDiscrete(x)
      val idx = z.mul(stride).sumNumber().longValue()
      Utils.features(Seq(idx), noFeatures)
    }
    (result, noFeatures)
  }
}
