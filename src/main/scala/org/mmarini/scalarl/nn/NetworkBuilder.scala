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

import org.yaml.snakeyaml.Yaml
import io.circe.Json
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.api.rng.Random

trait NetworkTopology {
  def nextLayer(layer: LayerBuilder): Option[LayerBuilder]
  def prevLayer(layer: LayerBuilder): Option[LayerBuilder]
}

case class NetworkBuilder(
  lossFunction: LossFunction,
  initializer:  Initializer,
  optimizer:    Optimizer,
  traceMode:    TraceMode,
  layers:       Array[LayerBuilder]) extends NetworkTopology {

  def nextLayer(layer: LayerBuilder): Option[LayerBuilder] = {
    val idx = layers.indexOf(layer)
    if (idx >= 0 && idx + 1 < layers.length) Some(layers(idx + 1)) else None
  }

  def prevLayer(layer: LayerBuilder): Option[LayerBuilder] = {
    val idx = layers.indexOf(layer)
    if (idx >= 1) Some(layers(idx - 1)) else None
  }

  def setNoInputs(noInputs: Int): NetworkBuilder = {
    val newLayers = InputLayerBuilder(noInputs) +: layers.tail
    copy(layers = newLayers)
  }

  def noInputs: Int = layers.head.asInstanceOf[InputLayerBuilder].noInputs

  def setLossFunction(lossFunction: LossFunction): NetworkBuilder = copy(lossFunction = lossFunction)
  def setInitializer(initializer: Initializer): NetworkBuilder = copy(initializer = initializer)
  def setOptimizer(optimizer: Optimizer): NetworkBuilder = copy(optimizer = optimizer)
  def setTraceMode(traceMode: TraceMode): NetworkBuilder = copy(traceMode = traceMode)
  def addLayers(layers: LayerBuilder*): NetworkBuilder = copy(layers = this.layers ++ layers)

  lazy val toJson: Json = Json.obj(
    "noInputs" -> Json.fromInt(noInputs),
    "lossFunction" -> lossFunction.toJson,
    "initializer" -> initializer.toJson,
    "optimizer" -> optimizer.toJson,
    "traceMode" -> traceMode.toJson,
    "layers" -> Json.arr(layers.map(_.toJson).toArray: _*))

  def buildProcessor: NetworkProcessor = {
    val clearTraceUpdaters = layers.map(_.buildClearTrace(this))
    val forwardUpdaters = layers.map(_.buildForward(this))
    val gradientUpdaters = layers.map(_.buildGradient(this))
    val deltaUpdaters = layers.map(_.buildDelta(this))
    val optimizerUpdaters = layers.map(_ => optimizer.buildOptimizer)
    val traceUpdaters = layers.map(_ => traceMode.buildTrace)
    val thetaUpdaters = layers.map(_ => UpdaterFactory.thetaUpdater)

    new NetworkProcessor(
      clearTraceUpdaters = clearTraceUpdaters,
      forwardUpdaters = forwardUpdaters,
      gradientUpdaters = gradientUpdaters,
      lossUpdater = lossFunction.buildGradient,
      deltaUpdaters = deltaUpdaters,
      optimizerUpdaters = optimizerUpdaters,
      traceUpdaters = traceUpdaters,
      thetaUpdaters = thetaUpdaters)
  }

  def buildData(random: Random): NetworkData = {
    val layerData = layers.map(_.buildData(this, initializer, random))
    new NetworkData(layerData)
  }
}

object NetworkBuilder {
  val DefaultAlpha = 0.1

  def apply(): NetworkBuilder = NetworkBuilder(
    lossFunction = MSELossFunction,
    initializer = XavierInitializer,
    optimizer = SGDOptimizer(alpha = DefaultAlpha),
    traceMode = NoneTraceMode,
    layers = Array(InputLayerBuilder(0)))

  def fromYaml(yaml: Yaml): NetworkBuilder = ???

}
