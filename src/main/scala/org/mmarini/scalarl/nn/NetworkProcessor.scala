package org.mmarini.scalarl.nn

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
  clearTraceUpdaters: Seq[Updater],
  forwardUpdaters:    Seq[Updater]) extends Network {

  /** Returns the network data with cleared eligibility traces */
  def clearTrace(data: NetworkData): NetworkData = {
    val newLayers = for {
      (updater, layerData) <- clearTraceUpdaters.zip(data.layers)
    } yield updater(layerData)
    data.copy(layers = newLayers)
  }

  /** Returns the data with computed outputs */
  def forward(data: NetworkData): NetworkData = {
    // Zip updaters with data layers
    val zipped = forwardUpdaters.zip(data.layers)

    // fold the zip to create the new sequence of layer data with outputs
    // and connecting the outputs to inputs of next layer
    val inputs = data.inputs.get
    val initialSeed = (Seq[LayerData](), inputs)
    val (newLayerData, _) = zipped.foldLeft(initialSeed) {
      case ((output, prevLayerData), (updater, data)) =>
        val newData = updater(data)
        (output :+ newData, newData("outputs"))
    }
    data.copy(layers = newLayerData)
  }

  /** Returns the data with changed parameters to fit the labels */
  def fit(data: NetworkData): NetworkData = {
    val withOutputs = forward(data)
    ???
  }

}