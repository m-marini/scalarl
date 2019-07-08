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
import org.nd4j.linalg.api.rng.Random

import io.circe.Json

trait NetworkTopology {
  def nextLayer(layer: LayerBuilder): Option[LayerBuilder]
  def prevLayer(layer: LayerBuilder): Option[LayerBuilder]
}

case class NetworkBuilder(
  lossFunction: LossFunction,
  initializer:  Initializer,
  optimizer:    Optimizer,
  traceMode:    TraceMode,
  normalizer:   Option[Normalizer],
  layers:       Seq[LayerBuilder]) extends NetworkTopology {

  def nextLayer(layer: LayerBuilder): Option[LayerBuilder] = {
    val idx = layers.indexOf(layer)
    if (idx >= 0 && idx + 1 < layers.length) Some(layers(idx + 1)) else None
  }

  def prevLayer(layer: LayerBuilder): Option[LayerBuilder] = {
    val idx = layers.indexOf(layer)
    if (idx >= 1) Some(layers(idx - 1)) else None
  }

  def setNoInputs(noInputs: Int): NetworkBuilder = {
    val newLayers = InputLayerBuilder("inputs", noInputs) +: layers.tail
    copy(layers = newLayers)
  }

  def setNormalizer(norm: Normalizer): NetworkBuilder = copy(normalizer = Some(norm))

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
    "layers" -> Json.arr(layers.tail.map(_.toJson).toArray: _*))

  private def internalForwardBuilder = {
    val normalized = normalizer.map(n =>
      OperationBuilder(data =>
        data + ("normalized" -> n.normalize(data("inputs"))))).getOrElse(
      OperationBuilder(data =>
        data + ("normalized" -> data("inputs"))))
    // Forwards updater of each layer
    val layerForwards = layers.map(layer =>
      layer.forwardBuilder(this))
    // merger updaters for each layer interconnection
    val in2Out = for {
      (prev, next) <- layers.zip(layers.tail)
    } yield {
      val toKey = s"${next.id}.inputs"
      val fromKey = s"${prev.id}.outputs"
      OperationBuilder(data =>
        data + (toKey -> data(fromKey)))
    }
    // Sequence the forward updaters and the mergers
    val seq = for {
      (forward, in2Out) <- layerForwards.zip(in2Out)
      builder <- Seq(forward, in2Out)
    } yield builder
    // Output Extractor
    val key = s"${layers.last.id}.outputs"
    val outputExractor = OperationBuilder(data =>
      data + ("outputs" -> data(key)))

    normalized +: seq :+ layerForwards.last :+ outputExractor
  }

  private def forwardBuilder =
    internalForwardBuilder.
      foldLeft(OperationBuilder())((acc, builder) =>
        acc.then(builder))

  private def backwardBuilder = {
    val reverseLayer = layers.reverse
    val layerBackward = reverseLayer.map(layer =>
      layer.deltaBuilder(this))

    // merger updaters for each layer interconnection
    val in2Out = for {
      (next, prev) <- reverseLayer.zip(reverseLayer.tail)
    } yield {
      val toKey = s"${prev.id}.delta"
      val fromKey = s"${next.id}.inputDelta"
      OperationBuilder(data =>
        data + (toKey -> data(fromKey)))
    }
    // Sequence the backword updaters and the mergers
    val seq = for {
      (deltaLayer, in2Out) <- layerBackward.zip(in2Out)
      builder <- Seq(deltaLayer, in2Out)
    } yield builder

    // deltaFeed
    val key = s"${reverseLayer.head.id}.delta"
    val deltaFeed = OperationBuilder(data =>
      data + (key -> data("delta")))

    deltaFeed +: seq :+ layerBackward.last
  }

  private def fitBuilder = {
    val gradient = layers.map(_.gradientBuilder(this))

    val optim = layers.map(layer =>
      optimizer.optimizeBuilder(layer.id))

    val trace = layers.map(layer =>
      layer.clearTraceBuilder(this))

    val brodcastDelta = layers.map(_.broadcastDeltaBuilder(this))

    val updated = layers.map(layer =>
      OperationBuilder.thetaBuilder(layer.id))

    val all = (internalForwardBuilder ++
      gradient :+
      lossFunction.deltaBuilder :+
      lossFunction.lossBuilder) ++
      backwardBuilder ++
      optim ++
      trace ++
      brodcastDelta ++
      updated

    all.foldLeft(OperationBuilder())((acc, builder) =>
      acc.then(builder))
  }

  def buildProcessor: NetworkProcessor = new NetworkProcessor(
    _forward = forwardBuilder.build,
    _fit = fitBuilder.build)

  def buildData(random: Random): NetworkData =
    layers.foldLeft(Map[String, INDArray]())((data, layer) =>
      data ++ layer.buildData(this, initializer, random))
}

object NetworkBuilder {
  val DefaultAlpha = 0.1

  def apply(): NetworkBuilder = NetworkBuilder(
    lossFunction = MSELossFunction,
    initializer = XavierInitializer,
    optimizer = SGDOptimizer(alpha = DefaultAlpha),
    traceMode = NoneTraceMode,
    normalizer = None,
    layers = Array(InputLayerBuilder("inputs", 0)))

  def fromJson(net: Json): NetworkBuilder = {
    val noInputs = net.hcursor.get[Int]("noInputs") match {
      case Right(x) => x
      case _        => throw new IllegalArgumentException("missing noInput in network definition")
    }
    val traceMode = net.hcursor.get[Json]("traceMode") match {
      case Right(x) => TraceMode.fromJson(x)
      case _        => throw new IllegalArgumentException("missing traceMode in network definition")
    }
    val optimizer = net.hcursor.get[Json]("optimizer") match {
      case Right(x) => Optimizer.fromJson(x)
      case _        => throw new IllegalArgumentException("missing optimizer in network definition")
    }
    val initializer = net.hcursor.get[Json]("initializer") match {
      case Right(x) => Initializer.fromJson(x)
      case _        => throw new IllegalArgumentException("missing initializer in network definition")
    }
    val lossFunction = net.hcursor.get[Json]("lossFunction") match {
      case Right(x) => LossFunction.fromJson(x)
      case _        => throw new IllegalArgumentException("missing lossFunction in network definition")
    }
    val layers = net.hcursor.get[Seq[Json]]("layers") match {
      case Right(x) => x.map(LayerBuilder.fromJson _)
      case _        => throw new IllegalArgumentException("missing layers in network definition")
    }
    NetworkBuilder().
      setNoInputs(noInputs).
      setTraceMode(traceMode).
      setOptimizer(optimizer).
      setLossFunction(lossFunction).
      setInitializer(initializer).
      addLayers(layers.toArray: _*)
  }
}
