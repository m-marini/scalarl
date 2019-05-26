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

/**
 * A eligibility trace network used to learn the Q function.
 * @constructor creates a new network
 * @param layers the layers of network from inputs to outputs
 * @param loss the loss function used to compute the error
 */
class TraceNetwork(
  val layers: Array[TraceLayer],
  val loss:   TraceLossFunction = LossFunctions.MSE) {

  /**
   * Returns the array of layer values from input to output
   * @parameters the input values
   */
  def forward(input: INDArray): Array[INDArray] = {
    val (result, _) = layers.foldLeft((Array(input), input)) {
      case ((result, input), layer) =>
        val out = layer.forward(input)
        (result :+ out, out)
    }
    result
  }

  /**
   * Returns the fit trace network and the loss value given the inputs, output labels and mask
   */
  def backward(input: INDArray, labels: INDArray, mask: INDArray): (TraceNetwork, Double) = {
    val activations = forward(input)
    // Transforms the layer activations into pair of input and output activations and layer
    val inOutLayers = activations.zip(activations.tail).zip(layers)
    // Computes the errors on the output layer
    val errors = loss.gradient(labels, activations.last, mask)
    // applies the backward propagation from output layer to input layer
    val (_, _, newLayers) = inOutLayers.reverse.foldLeft((errors, mask, Seq[TraceLayer]())) {
      case ((error, mask, layers), ((in, out), layer)) => {
        val (newLayer, backErrors, backMask) = layer.backward(in, out, error, mask)
        (backErrors, backMask, newLayer +: layers)
      }
    }
    (
      new TraceNetwork(layers = newLayers.toArray, loss),
      loss(labels, activations.last, mask))
  }

  def clearTraces(): TraceNetwork = {
    layers.foreach(_.clearTraces())
    this
  }
}
