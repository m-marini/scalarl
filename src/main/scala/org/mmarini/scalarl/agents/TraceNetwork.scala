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

class TraceNetwork(
  val layers: Array[TraceLayer],
  val loss:   TraceLossFunction = LossFunctions.MSE) {

  def forward(input: INDArray): Array[INDArray] = {
    val (result, _) = layers.foldLeft((Array(input), input)) {
      case ((result, input), layer) =>
        val out = layer.forward(input)
        (result :+ out, out)
    }
    result
  }

  /**
   * Returns the loss value and update the network fitting the labels with mask given an input
   */
  def backward(input: INDArray, labels: INDArray, mask: INDArray): Double = {
    val activations = forward(input)
    val inOut = activations.zip(activations.tail).zip(layers)
    val errors = loss.gradient(labels, activations.last, mask)
    inOut.reverse.foldLeft((errors, mask)) {
      case ((error, mask), ((in, out), layer)) =>
        layer.backward(in, out, error, mask)
    }
    loss(labels, activations.last, mask)
  }

  def clearTraces(): TraceNetwork = {
    layers.foreach(_.clearTraces())
    this
  }
}
