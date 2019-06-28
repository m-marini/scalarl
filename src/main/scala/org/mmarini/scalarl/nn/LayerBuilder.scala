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

  /** Returns the number of inputs */
  def noInputs(topology: NetworkTopology): Int = topology.prevLayer(this).map(_.noOutputs(topology)).getOrElse(0)

  /** Returns the number of outputs */
  def noOutputs(topology: NetworkTopology): Int

  /** Returns the updater that clears the eligibility traces of the layer */
  def buildClearTrace(topology: NetworkTopology): Updater

  /** Returns the updater that forwards the inputs */
  def buildForward(topology: NetworkTopology): Updater

  /** Returns the updater that computes the gradient */
  def buildGradient(topology: NetworkTopology): Updater

  /** Returns the updater that computes the delta by backwording errors */
  def buildDelta(topology: NetworkTopology): Updater

  /** Returns the layer data for the [[LayerBuilder]] */
  def buildData(topology: NetworkTopology, initializer: Initializer, random: Random): LayerData

  /** Returns the json representation of layer */
  def toJson: Json
}

case class InputLayerBuilder(noInputs: Int) extends LayerBuilder {

  /** Returns the number of inputs */
  override def noInputs(topology: NetworkTopology): Int = noInputs

  /** Returns the number of outputs */
  override def noOutputs(topology: NetworkTopology): Int = noInputs

  /** Returns the updater that clears the eligibility traces of the layer */
  override def buildClearTrace(topology: NetworkTopology): Updater = UpdaterFactory.identityUpdater

  /** Returns the updater that forwards the inputs */
  override def buildForward(topology: NetworkTopology): Updater = (data: LayerData) =>
    data + ("outputs" -> data("inputs"))

  /** Returns the updater that computes the gradient */
  override def buildGradient(topology: NetworkTopology): Updater = UpdaterFactory.identityUpdater

  /** Returns the updater that computes the delta by backwording errors */
  override def buildDelta(topology: NetworkTopology): Updater = UpdaterFactory.identityUpdater

  /** Returns the layer data for the [[LayerBuilder]] */
  override def buildData(topology: NetworkTopology, initializer: Initializer, random: Random): LayerData = Map()

  /** Returns the json representation of layer */
  override def toJson: Json = Json.Null
}

/**
 * Defines the activation layer architecture and build the layer functional updater such that for clear trace.
 *
 * @constructor Creates an activation layer
 */
case class ActivationLayerBuilder(activation: ActivationFunction) extends LayerBuilder {

  def noOutputs(topology: NetworkTopology): Int = noInputs(topology)

  def buildClearTrace(context: NetworkTopology): Updater = UpdaterFactory.identityUpdater

  def buildGradient(topology: NetworkTopology): Updater = UpdaterFactory.identityUpdater

  def buildForward(context: NetworkTopology): Updater = activation.buildActivation

  def buildDelta(context: NetworkTopology): Updater = activation.buildDelta

  def buildData(topology: NetworkTopology, initializer: Initializer, random: Random): LayerData = Map()

  lazy val toJson = Json.obj(
    "type" -> Json.fromString("ACTIVATION"),
    "activation" -> activation.toJson)
}

/**
 * Defines the activation layer architecture and build the layer functional updater such that for clear trace.
 */
case class DenseLayerBuilder(noOutputs: Int) extends LayerBuilder {

  def noOutputs(topology: NetworkTopology): Int = noOutputs

  def buildClearTrace(context: NetworkTopology): Updater = {
    val n = noInputs(context)
    val zeroTrace = Nd4j.zeros((n + 1) * noOutputs)
    (data: LayerData) => data + ("trace" -> zeroTrace)
  }

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

  def buildForward(topology: NetworkTopology): Updater = {
    val fw = weights(topology)
    val fb = bias(topology)

    // Creates the updater
    (data: LayerData) => {
      val inputs = data("inputs")
      val theta = data("theta")
      val w = fw(theta)
      val b = fb(theta)
      val y = inputs.mmul(w).addi(b)
      data + ("outputs" -> y)
    }
  }

  def buildGradient(topology: NetworkTopology): Updater = {
    val n = noInputs(topology)
    val m = noOutputs
    val bGrad = Nd4j.ones(m)

    // Creates the updater
    (data: LayerData) => {
      val inputs = data("inputs")
      val wGrad = inputs.transpose().broadcast(n, m)
      val wFlatten = wGrad.ravel()
      val grad = Nd4j.hstack(wFlatten, bGrad)
      data + ("gradient" -> grad)
    }
  }

  def buildDelta(topology: NetworkTopology): Updater = {
    val fw = weights(topology)

    // Creates the updater
    (data: LayerData) => {
      val delta = data("delta")
      val theta = data("theta")
      val w = fw(theta)
      val inpDelta = delta.mmul(w.transpose())
      data + ("inputDelta" -> inpDelta)
    }
  }

  def buildData(topology: NetworkTopology, initializer: Initializer, random: Random): LayerData = {
    val n = noInputs(topology)
    val m = noOutputs
    val weights = initializer.build(n, m, random)
    val bias = Nd4j.zeros(m)
    val theta = Nd4j.hstack(weights.ravel(), bias)
    val zeros = Nd4j.zeros(n * m + m)
    Map(
      "theta" -> theta,
      "trace" -> zeros,
      "m1" -> zeros,
      "m2" -> zeros)
  }

  lazy val toJson = Json.obj(
    "type" -> Json.fromString("DENSE"),
    "noOutputs" -> Json.fromInt(noOutputs))
}
