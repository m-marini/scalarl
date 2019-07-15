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
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.indexing.NDArrayIndex

import io.circe.Json

/**
 * Defines layer architecture and builds the layer functional updater such that for clear trace.
 *
 *  - [[ActivationLayerBuilder]] defines the activation network layer
 *  - [[DenseLayerBuilder]] defines the dense network layer
 *
 *
 */
trait LayerBuilder {
  def id: String

  /** Returns the number of inputs */
  def noInputs(topology: NetworkTopology): Int = topology.prevLayer(this).map(_.noOutputs(topology)).getOrElse(0)

  /** Returns the number of outputs */
  def noOutputs(topology: NetworkTopology): Int

  /** Returns the updater that forwards the inputs */
  def forwardBuilder(topology: NetworkTopology): OperationBuilder

  /** Returns the updater that computes the gradient */
  def gradientBuilder(topology: NetworkTopology): OperationBuilder

  /** Returns the updater that computes the delta by backwording errors */
  def deltaBuilder(topology: NetworkTopology): OperationBuilder

  /** Returns the operation that broadcast delta to theta */
  def broadcastDeltaBuilder(topology: NetworkTopology): OperationBuilder

  /** Returns the layer data for the [[LayerBuilder]] */
  def buildData(topology: NetworkTopology, initializer: Initializer, random: Random): NetworkData

  /** Returns the json representation of layer */
  def toJson: Json
}

trait KeyBuilder {
  def id: String

  def key(name: String): String = s"${id}.${name}"
}

case class InputLayerBuilder(id: String, noInputs: Int) extends LayerBuilder with KeyBuilder {

  override def noInputs(topology: NetworkTopology): Int = noInputs

  override def noOutputs(topology: NetworkTopology): Int = noInputs

  override def forwardBuilder(topology: NetworkTopology): OperationBuilder = OperationBuilder(data =>
    data + (key("outputs") -> data("normalized")))

  override def gradientBuilder(topology: NetworkTopology): OperationBuilder = OperationBuilder()

  override def deltaBuilder(topology: NetworkTopology): OperationBuilder = OperationBuilder()

  override def broadcastDeltaBuilder(topology: NetworkTopology): OperationBuilder = OperationBuilder()

  override def buildData(topology: NetworkTopology, initializer: Initializer, random: Random): NetworkData = Map()

  override def toJson: Json = Json.Null
}

/**
 * Defines the activation layer architecture and build the layer functional updater such that for clear trace.
 *
 * @constructor Creates an activation layer
 */
case class ActivationLayerBuilder(id: String, activation: ActivationFunction) extends LayerBuilder with KeyBuilder {
  def noOutputs(topology: NetworkTopology): Int = noInputs(topology)

  def gradientBuilder(topology: NetworkTopology): OperationBuilder = OperationBuilder()

  override def forwardBuilder(context: NetworkTopology): OperationBuilder = OperationBuilder(data => {
    val inputs = data(key("inputs"))
    val outputs = activation.activate(inputs)
    data + (key("outputs") -> outputs)
  })

  def deltaBuilder(context: NetworkTopology): OperationBuilder = OperationBuilder(data => {
    val inputs = data(key("inputs"))
    val outputs = data(key("outputs"))
    val delta = data(key("delta"))
    val inputDelta = activation.inputDelta(inputs, outputs, delta)
    data + (key("inputDelta") -> inputDelta)
  })

  override def broadcastDeltaBuilder(topology: NetworkTopology): OperationBuilder = OperationBuilder()

  def buildData(topology: NetworkTopology, initializer: Initializer, random: Random): NetworkData = Map()

  lazy val toJson = Json.obj(
    "id" -> Json.fromString(id),
    "type" -> Json.fromString("ACTIVATION"),
    "activation" -> activation.toJson)
}

/**
 * Defines the activation layer architecture and build the layer functional updater such that for clear trace.
 */
case class DenseLayerBuilder(id: String, noOutputs: Int) extends LayerBuilder with KeyBuilder {
  def noOutputs(topology: NetworkTopology): Int = noOutputs

  /** Returns the converter og thetas to weights */
  def weights(topology: NetworkTopology): INDArray => INDArray = {
    val n = noInputs(topology)
    val m = noOutputs
    (theta: INDArray) => theta.get(NDArrayIndex.interval(0, n * m)).reshape(n, m)
  }

  /** Returns the converter of thetas to bias */
  def bias(topology: NetworkTopology): INDArray => INDArray = {
    val n = noInputs(topology)
    val m = noOutputs
    (theta: INDArray) => theta.get(NDArrayIndex.interval(n * m, n * (m + 1)))
  }

  override def forwardBuilder(topology: NetworkTopology): OperationBuilder = {
    val fw = weights(topology)
    val fb = bias(topology)

    // Creates the updater
    OperationBuilder(data => {
      val inputs = data(key("inputs"))
      val theta = data(key("theta"))
      val w = fw(theta)
      val b = fb(theta)
      val y = inputs.mmul(w).addi(b)
      data + (key("outputs") -> y)
    })
  }

  def gradientBuilder(topology: NetworkTopology): OperationBuilder = {
    val n = noInputs(topology)
    val m = noOutputs
    val bGrad = Nd4j.ones(m)

    // Creates the updater
    OperationBuilder(data => {
      val inputs = data(key("inputs"))
      val wGrad = inputs.transpose().broadcast(n, m)
      val wFlatten = wGrad.ravel()
      val grad = Nd4j.hstack(wFlatten, bGrad)
      data + (key("gradient") -> grad)
    })
  }

  def deltaBuilder(topology: NetworkTopology): OperationBuilder = {
    val fw = weights(topology)

    // Creates the updater
    OperationBuilder(data => {
      val delta = data(key("delta"))
      val theta = data(key("theta"))
      val w = fw(theta)
      val inpDelta = delta.mmul(w.transpose())
      data + (key("inputDelta") -> inpDelta)
    })
  }

  override def broadcastDeltaBuilder(topology: NetworkTopology): OperationBuilder = {
    val n = noInputs(topology)
    val m = noOutputs
    OperationBuilder(data => {
      val delta = data(key("delta"))
      val wDelta = delta.broadcast(n, m)
      val bDelta = delta
      val thetaDelta = Nd4j.hstack(wDelta.ravel(), bDelta)
      data + (key("thetaDelta") -> thetaDelta)
    })
  }

  def buildData(topology: NetworkTopology, initializer: Initializer, random: Random): NetworkData = {
    val n = noInputs(topology)
    val m = noOutputs
    val weights = initializer.build(n, m, random)
    val bias = Nd4j.zeros(m)
    val theta = Nd4j.hstack(weights.ravel(), bias)
    val zeros = Nd4j.zeros(n * m + m)
    Map(
      key("theta") -> theta,
      key("trace") -> zeros,
      key("m1") -> zeros,
      key("m2") -> zeros)
  }

  lazy val toJson = Json.obj(
    "id" -> Json.fromString(id),
    "type" -> Json.fromString("DENSE"),
    "noOutputs" -> Json.fromInt(noOutputs))
}

object LayerBuilder {
  def fromJson(json: Json): LayerBuilder =
    json.hcursor.get[String]("type") match {
      case Right("DENSE")      => denseFromJson(json)
      case Right("ACTIVATION") => activationFromJson(json)
      case Right(x)            => throw new IllegalArgumentException(s"""layer type "${x}" illegal""")
      case Left(x)             => throw new IllegalArgumentException("missing layer type")
    }

  private def denseFromJson(json: Json) = {
    val id = json.hcursor.get[String]("id") match {
      case Right(x) => x
      case Left(x)  => throw new IllegalArgumentException("missing layer id")
    }
    val noOutputs = json.hcursor.get[Int]("noOutputs") match {
      case Right(x) => x
      case Left(x)  => throw new IllegalArgumentException("missing noOutputs")
    }
    DenseLayerBuilder(id, noOutputs)
  }

  private def activationFromJson(json: Json) = {
    val id = json.hcursor.get[String]("id") match {
      case Right(x) => x
      case Left(x)  => throw new IllegalArgumentException("missing layer id")
    }
    val activation = json.hcursor.get[String]("activation") match {
      case Right("TANH") => TanhActivationFunction
      case Right(x)      => throw new IllegalArgumentException(s"""activation "${x}" illegal""")
      case _             => throw new IllegalArgumentException("missing activation")
    }
    ActivationLayerBuilder(id, activation)
  }
}