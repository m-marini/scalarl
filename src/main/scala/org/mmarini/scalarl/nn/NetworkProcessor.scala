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

/**
 * Processes the inputs to produce the outputs and fits the network parameters to fit for the expected result.
 *
 * Use [[NetworkBuilder]] to build a NetworkProcessor
 *
 * @example
 * The forward method generates the outputs
 * {{{
 *   val networkProcessor: NetworkProcessor  = ...
 *   val initialNetworkData: NetworkData = ...
 *   val inputs: INDArray = ...
 *
 *   val inputNetworkData: NetworkData = initialNetworkData.setInputs(inputs)
 *   val outputNetworkData:  NetworkData = networkProcessor.forward(inputNetworkData)
 *   val outputs: Option[INDArray] = outputNetworkData.outputs
 * }}}
 *
 * The fit method changes the network data fitting for specific expectations
 * {{{
 *   val networkProcessor: NetworkProcessor  = ...
 *   val initialNetworkData: NetworkData = ...
 *   val inputs: INDArray = ...
 *   val labels: INDArray = ...
 *   val mask: INDArray = ...
 *
 *   val inputNetworkData: NetworkData = initialNetworkData.setInputs(inputs).setLabels(labels, mask)
 *   val fittedNetworkData:  NetworkData = networkProcessor.fit(inputNetworkData)
 * }}}
 */
class NetworkProcessor(
                        _forward: Operation,
                        _fit: Operation)
  extends Network {

  /** Returns the data with computed outputs */
  def forward(data: NetworkData, inputs: INDArray): INDArray = {
    Sentinel(inputs, "inputs are numbers")

    _forward(data +
      ("inputs" -> inputs))("outputs")
  }

  /** Returns the data with changed parameters to fit the labels */
  def fit(
           data: NetworkData,
           inputs: INDArray,
           labels: INDArray,
           mask: INDArray,
           noClearTrace: INDArray): NetworkData = {
    Sentinel(inputs, "inputs")
    Sentinel(labels, "labels")
    Sentinel(mask, "mask")

    _fit(data +
      ("inputs" -> inputs) +
      ("labels" -> labels) +
      ("mask" -> mask) +
      ("noClearTrace" -> noClearTrace))
  }
}
