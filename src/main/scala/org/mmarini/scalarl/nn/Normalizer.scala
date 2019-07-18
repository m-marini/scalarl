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

package org.mmarini.scalarl.nn

import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j

import io.circe.Json

/**
 * Defines layer architecture and builds the layer functional updater such that for clear trace.
 *
 *  - [[ActivationLayerBuilder]] defines the activation network layer
 *  - [[DenseLayerBuilder]] defines the dense network layer
 *
 *
 */
trait Normalizer {

  /** Returns the normalized value */
  def normalize(x: INDArray): INDArray

  /** Returns the json representation of layer */
  def toJson: Json
}

case class LinearNormalizer(offset: INDArray, scale: INDArray) extends Normalizer {
  Sentinel(offset, "offset")
  Sentinel(scale, "scale")

  def normalize(x: INDArray): INDArray =
    x.add(offset).muli(scale)

  def toJson: Json = {
    val o = offset.toDoubleVector().map(x => Json.fromDouble(x).get)
    val s = scale.toDoubleVector().map(x => Json.fromDouble(x).get)
    Json.obj(
      "type" -> Json.fromString("LINEAR"),
      "offset" -> Json.arr(o: _*),
      "scale" -> Json.arr(s: _*))
  }
}

object Normalizer {
  def fromJson(json: Json): LinearNormalizer =
    json.hcursor.get[String]("type") match {
      case Right("LINEAR") =>
        val offset = json.hcursor.get[Array[Double]]("offset").right.get
        val scala = json.hcursor.get[Array[Double]]("scale").right.get
        LinearNormalizer(
          offset = Nd4j.create(offset),
          scale = Nd4j.create(scala))
      case Right(x) => throw new IllegalArgumentException(s"""normalizer type "${x}" illegal""")
      case Left(x)  => throw new IllegalArgumentException("missing normalizer type")
    }

  def minMax(min: INDArray, max: INDArray) = {
    val offset = max.add(min).negi().divi(2.0)
    val scale = Nd4j.ones(min.shape(): _*).mul(2.0).divi(max.sub(min))
    LinearNormalizer(offset, scale)
  }

  def minMax(nodes: Int, min: Double, max: Double): LinearNormalizer = {
    val offset = Nd4j.ones(nodes).mul(-(min + max) / 2.0)
    val scale = Nd4j.ones(nodes).mul(2.0 / (max - min))
    LinearNormalizer(offset, scale)
  }
}