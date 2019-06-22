
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
  clearTraceUpdaters: Array[Updater],
  forwardUpdaters:    Array[Updater],
  gradientUpdaters:   Array[Updater],
  deltaUpdaters:      Array[Updater],
  optimizerUpdaters:  Array[Updater],
  traceUpdaters:      Array[Updater],
  thetaUpdaters:      Array[Updater]) extends Network {

  type Reducer = (LayerData, LayerData) => LayerData

  /** Returns the network data with cleared eligibility traces */
  def clearTrace(data: NetworkData): NetworkData =
    concurrentPass(clearTraceUpdaters)(data)

  def forwardReducer(left: LayerData, right: LayerData) = right + ("inputs" -> left("outputs"))

  /** Returns the data with computed outputs */
  def forward(data: NetworkData): NetworkData =
    forwardPass(
      forwardUpdaters,
      forwardReducer)(data)

  /**
   * Returns the new [[NetworkData]]
   * executing updaters on all layers
   */
  def concurrentPass(updatersLayer: Array[Updater])(data: NetworkData): NetworkData = {
    val outLayers = for {
      (updater, data) <- updatersLayer.zip(data.layers)
    } yield updater(data)
    data.copy(layers = outLayers)
  }

  def forwardPass(updaters: Array[Updater], reducer: Reducer)(data: NetworkData): NetworkData = {
    val zip = updaters.zip(data.layers)
    val seed = (Seq[LayerData](), Map().asInstanceOf[LayerData])
    val (newLayerData, _) = zip.foldLeft(seed) {
      case ((out, in), (updater, layer)) =>
        val newLayer = updater(reducer(in, layer))
        (out :+ newLayer, newLayer)
    }
    data.copy(layers = newLayerData.toArray)
  }

  def backwardPass(updaters: Array[Updater], reducer: Reducer)(data: NetworkData): NetworkData = {
    val zip = updaters.zip(data.layers)
    val seed = (Seq[LayerData](), Map().asInstanceOf[LayerData])
    val (newLayerData, _) = zip.foldRight(seed) {
      case ((updater, layer), (out, in)) =>
        val newLayer = updater(reducer(in, layer))
        (newLayer +: out, newLayer)
    }
    data.copy(layers = newLayerData.toArray)
  }

  def maskReducer(left: LayerData, right: LayerData) = left + ("inputMask" -> right("mask"))
  def deltaReducer(left: LayerData, right: LayerData) = left + ("delta" -> right("delta"))

  /**
   * Returns the data with changed parameters to fit the labels
   * The process of bacward consists of
   * Sequentialization
   * {{{
   * |------------------------------|--------------------------|
   * | forward compute outputs      | backward compute mask    |
   * | concurrent compute gradients |                          |
   * |------------------------------|--------------------------|
   * | backward compute delta       | concurrent compute optim |
   * |                              | concurrent compute trace |
   * |------------------------------|--------------------------|
   * | concurrent update theta      |                          |
   * |------------------------------|--------------------------|
   * }}}
   */
  override def fit(data: NetworkData): NetworkData = {
    val withOutputs = forward(data)
    val withGradient = concurrentPass(gradientUpdaters)(withOutputs)

    val withDelta = backwardPass(deltaUpdaters, deltaReducer)(withGradient)
    val withOptim = concurrentPass(optimizerUpdaters)(withDelta)
    val withTrace = concurrentPass(traceUpdaters)(withOptim)

    val updated = concurrentPass(thetaUpdaters)(withOptim)

    updated
  }

}