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

package org.mmarini.scalarl.agents

import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.indexing.NDArrayIndex

/**
 */
class TraceDenseLayer(weights: INDArray, bias: INDArray, traces: INDArray) {

  /** Returns the output of layer given an input */
  def forward(input: INDArray): INDArray = {
    val z = input.mmul(weights)
    val bb = bias.broadcast(z.shape(): _*)
    val y = z.add(bb)
    y
  }

  /** Returns the gradient of weights and bias given the input and output of layer */
  def gradient(data: (INDArray, INDArray)): (INDArray, INDArray) = data match {
    case (input, output) =>
      val n = input.size(0)
      require(n == output.size(0))
      val ni = input.size(1)
      val no = output.size(1)
      val bGrad = Nd4j.ones(n, no)
      val wGrad1 = Nd4j.zeros(n, ni, 1)
      wGrad1.put(Array(NDArrayIndex.all, NDArrayIndex.all, NDArrayIndex.point(0)), input)
      val wGrad = wGrad1.broadcast(n, ni, no)
      (wGrad, bGrad)
  }

  /**
   * Returns the layer by updating traces given input and output of layer
   */
  def updateTraces(data: (INDArray, INDArray)): TraceDenseLayer = data match {
    case (input, output) =>
      val (wGrad, bGrad) = gradient((input, output))
      this
  }

  /**
   * Returns the input error given the input, output and output error of the layer
   *
   * Updates the layer parameters applying the gradient descendant algorithm
   */
  def backward(data: (INDArray, INDArray, INDArray)): INDArray = data match {
    case (input, output, error) =>
      updateTraces((input, output))
       ???
  }

}
