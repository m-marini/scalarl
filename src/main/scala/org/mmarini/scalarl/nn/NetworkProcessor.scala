
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
  _forward: Updater,
  _fit:     Updater)
  extends Network {

  /** Returns the data with computed outputs */
  def forward(data: NetworkData, inputs: INDArray): INDArray = _forward(data +
    ("inputs" -> inputs))("outputs")

  /** Returns the data with changed parameters to fit the labels */
  def fit(
    data:         NetworkData,
    inputs:       INDArray,
    labels:       INDArray,
    mask:         INDArray,
    noClearTrace: INDArray): NetworkData = _fit(data +
    ("inputs" -> inputs) +
    ("labels" -> labels) +
    ("mask" -> mask) +
    ("noClearTrace" -> noClearTrace))
}