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

import io.circe.Json
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.indexing.NDArrayIndex

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

  /** Returns the json representation of layer */
  def toJson: Json
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

  lazy val toJson = Json.obj(
    "type" -> Json.fromString("DENSE"),
    "noOutputs" -> Json.fromInt(noOutputs))
}
