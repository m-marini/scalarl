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
import org.mmarini.scalarl.v4.Utils._
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j._

object Normalizer {
  /**
   *
   * @param conf
   * @param noDims
   * @return
   */
  def fromJson(conf: ACursor)(noDims: Int): INDArray => INDArray = {
    val clipMin = create(conf.get[Array[Double]]("clipMin").toTry.get)
    require(clipMin.shape() sameElements Array(1L, noDims))
    val clipMax = create(conf.get[Array[Double]]("clipMax").toTry.get)
    require(clipMax.shape() sameElements Array(1L, noDims))
    val offset = create(conf.get[Array[Double]]("offset").toTry.get)
    require(offset.shape() sameElements Array(1L, noDims))
    val max = create(conf.get[Array[Double]]("max").toTry.get)
    require(max.shape() sameElements Array(1, noDims))
    val scale = ones(noDims).divi(max.sub(offset))
    Normalizer(offset, scale, clipMin, clipMax)
  }

  def apply(offset: INDArray, scale: INDArray, clipMin: INDArray, clipMax: INDArray): INDArray => INDArray =
    x => clip(x, clipMin, clipMax, true).subi(offset).muli(scale)
}