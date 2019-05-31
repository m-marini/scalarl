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
case class TraceDenseLayer(
  val weights:      INDArray,
  val bias:         INDArray,
  val weightTraces: INDArray,
  val biasTraces:   INDArray,
  val gamma:        Double,
  val lambda:       Double,
  val learningRate: Double,
  val traceUpdater: TraceUpdater) extends TraceLayer {

  /** Returns the output of layer given an input */
  override def forward(input: INDArray): INDArray = {
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
   * Returns a new layer by updating traces given input, output and output mask of layer
   */
  override def clearTraces(): TraceLayer =
    copy(
      weightTraces = Nd4j.zeros(weightTraces.shape(): _*),
      biasTraces = Nd4j.zeros(biasTraces.shape(): _*))

  /**
   * Returns the backward errors and mask after updating the layer parameters given the input, output, output, errors
   * and output mask
   */
  override def backward(input: INDArray, output: INDArray, errors: INDArray, mask: INDArray): (TraceLayer, INDArray, INDArray) = {

    val (newWeightTraces, newBiasTraces) = updatedTraces(input, output, mask)

    val ni = weights.size(0)
    val no = weights.size(1)

    val dWeights = newWeightTraces.mul(learningRate).muli(errors.broadcast(ni, no))
    val dBias = newBiasTraces.mul(learningRate).muli(errors)

    val inpErrors = errors.mmul(weights.transpose())

    val newWeights = weights.add(dWeights)
    val newBias = bias.add(dBias)
    val newLayer = copy(
      bias = newBias,
      weights = newWeights,
      biasTraces = newBiasTraces,
      weightTraces = newWeightTraces)

    val inpMask = Nd4j.ones(input.shape(): _*)
    (newLayer, inpErrors, inpMask)
  }

  /**
   * Returns the updated traces for input, output and output mask
   */
  private def updatedTraces(input: INDArray, output: INDArray, mask: INDArray): (INDArray, INDArray) = {
    val (wGrad, bGrad) = gradient(input, output)
    wGrad.muli(mask.broadcast(wGrad.shape(): _*))
    bGrad.muli(mask)
    traceUpdater(
      weightTraces.mul(lambda * gamma),
      biasTraces.mul(lambda * gamma),
      wGrad, bGrad)
    //    val newWeightTraces = weightTraces.mul(lambda * gamma).addi(wGrad)
    //    val newBiasTraces = biasTraces.mul(lambda * gamma).addi(bGrad)
    //    (newWeightTraces, newBiasTraces)
  }
}

object TraceDenseLayer {
  def apply(
    weights:      INDArray,
    bias:         INDArray,
    gamma:        Double,
    lambda:       Double,
    learningRate: Double,
    traceUpdater: TraceUpdater): TraceDenseLayer = {
    val wTraces = Nd4j.zeros(weights.shape(): _*)
    val bTraces = Nd4j.zeros(bias.shape(): _*)
    new TraceDenseLayer(
      weights = weights,
      bias = bias,
      weightTraces = wTraces,
      biasTraces = bTraces,
      gamma = gamma,
      lambda = lambda,
      learningRate = learningRate,
      traceUpdater = traceUpdater)
  }

  def apply(
    noInputs:     Long,
    noOutputs:    Long,
    gamma:        Double,
    lambda:       Double,
    learningRate: Double,
    traceUpdater: TraceUpdater): TraceDenseLayer = {
    // Xavier initialization
    val weights = Nd4j.randn(noInputs, noOutputs).muli(2.0 / (noInputs + noOutputs))
    val bias = Nd4j.zeros(1L, noOutputs)
    TraceDenseLayer(
      weights = weights,
      bias = bias,
      gamma = gamma,
      lambda = lambda,
      learningRate = learningRate,
      traceUpdater = traceUpdater)
  }
}
