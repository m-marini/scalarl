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

import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.api.ndarray.INDArray

class DenseLayer(
  config:    Config,
  noInputs:  Int,
  noOutputs: Int) extends Layer {

  /** Generate new LayerData containing the computed output for the inputData */
  def forwardPass(data: LayerData): LayerData = {
    val inputs = data("inputs")
    val parms = data("parms")
    val weights = mapWeights(parms)
    val bias = mapBias(parms)
    val z = inputs.mmul(weights)
    val y = z.add(bias)
    data + ("outputs" -> y)
  }

  def clearTraces(data: LayerData): LayerData = {
    val traces = Nd4j.zeros(noInputs * (noOutputs + 1))
    data + ("traces" -> traces)
  }

  def backwardPass(data: LayerData): LayerData = {
    val errors = data("errors")
    val parms = data("parms")
    val inputs = data("inputs")
    val weights = mapWeights(parms)
    val inpErrors = errors.mmul(weights.transpose())
    val inpMask = Nd4j.ones(inputs.shape(): _*)
    data + ("inputErrors" -> inpErrors) + ("inputMask" -> inpMask)
  }

  /** Returns the layer data with optimized learning parameters */
  def optimizePass(data: LayerData): LayerData = {
    config.optimzer.map(_(this)(data)).getOrElse(data)
  }

  /** Returns the layer data with parameter changes */
  def updatePass(data: LayerData): LayerData = {
    config.updater.map(_(this)(data)).getOrElse(data)
  }

  def mapWeights(data: INDArray): INDArray = {
    ???
  }

  def mapBias(data: INDArray): INDArray = {
    ???
  }

  def mapParms(weights: INDArray, bias: INDArray): INDArray =
    Nd4j.vstack(weights, bias).ravel()

  def gradientPass(data: LayerData): LayerData = {
    val inputs = data("inputs")
    val bGrad = Nd4j.ones(1, noOutputs).ravel();
    val wGrad = inputs.transpose().broadcast(noInputs, noOutputs).ravel();
    val gradient = mapParms(wGrad, bGrad)
    data + ("gradient" -> gradient)
  }
}
