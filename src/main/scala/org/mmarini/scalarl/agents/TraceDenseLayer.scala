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
import org.nd4j.linalg.indexing.INDArrayIndex

/**
 */
class TraceDenseLayer(
  val weights:      INDArray,
  val bias:         INDArray,
  val weightTraces: INDArray,
  val biasTraces:   INDArray,
  val gamma:        Double,
  val lambda:       Double,
  val learningRate: Double) {

  /** Returns the output of layer given an input */
  def forward(input: INDArray): INDArray = {
    val z = input.mmul(weights)
    val y = z.add(bias)
    y
  }

  /** Returns the gradient of weights and bias given the input and output of layer */
  def gradient(input: INDArray, output: INDArray): (INDArray, INDArray) = {
    val ni = input.size(1)
    val no = output.size(1)
    val bGrad = Nd4j.ones(1, no)
    val wGrad = input.transpose().broadcast(ni, no)
    (wGrad, bGrad)
  }

  /**
   * Returns the layer by updating traces given input, output and output mask of layer
   */
  def clearTraces(): TraceDenseLayer = {
    weightTraces.put(Array(NDArrayIndex.all), 0.0)
    biasTraces.put(Array(NDArrayIndex.all), 0.0)
    this
  }

  /**
   * Returns the layer by updating traces given input, output and output mask of layer
   */
  def updateTraces(input: INDArray, output: INDArray, mask: INDArray): TraceDenseLayer = {
    weightTraces.muli(lambda * gamma)
    biasTraces.muli(lambda * gamma)
    val (wGrad, bGrad) = gradient(input, output)
    wGrad.muli(mask.broadcast(wGrad.shape(): _*))
    bGrad.muli(mask)
    weightTraces.addi(wGrad)
    biasTraces.addi(bGrad)
    this
  }

  /**
   * Returns the backward errors after updating the layer parameters given the input, output, output, errors
   * and output mask
   */
  def backward(input: INDArray, output: INDArray, errors: INDArray, mask: INDArray): INDArray = {
    ???
  }
}

object TraceDenseLayer {
  def apply(
    weights:      INDArray,
    bias:         INDArray,
    gamma:        Double,
    lambda:       Double,
    learningRate: Double): TraceDenseLayer = {
    val wTraces = Nd4j.zeros(weights.shape(): _*)
    val bTraces = Nd4j.zeros(bias.shape(): _*)
    new TraceDenseLayer(
      weights = weights,
      bias = bias,
      weightTraces = wTraces,
      biasTraces = bTraces,
      gamma = gamma,
      lambda = lambda,
      learningRate = learningRate)
  }
}
