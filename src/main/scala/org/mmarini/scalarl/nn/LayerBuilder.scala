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

/**
 * Defines layer architecture and builds the layer functional updater such that for clear trace.
 *
 *  - [[ActivationLayerBuilder]] defines the activation network layer
 *  - [[DenseLayerBuilder]] defines the dense network layer
 *
 *
 */
trait LayerBuilder {

  /** Returns the updater that clears the eligibility traces of the layer */
  def buildClearTrace(context: NetworkBuilder): Updater

  /** Returns the updater that forward the inputs */
  def buildForward(context: NetworkBuilder): Updater

  /** Returns the json rappresentation of layer */
  def toJson: Json
}

/**
 * Defines the activation layer architecture and build the layer functional updater such that for clear trace.
 *
 * @constructor Creates an activation layer
 */
case class ActivationLayerBuilder(activation: ActivationFunction) extends LayerBuilder {

  override def buildClearTrace(context: NetworkBuilder): Updater = UpdaterFactory.identityUpdater

  def buildForward(context: NetworkBuilder): Updater = ???

  lazy val toJson = Json.obj(
    "type" -> Json.fromString("ACTIVATION"),
    "activation" -> activation.toJson)
}

/**
 * Defines the activation layer architecture and build the layer functional updater such that for clear trace.
 */
case class DenseLayerBuilder(noOutputs: Int) extends LayerBuilder {
  def noInputs(context: NetworkBuilder): Int = ???

  def buildClearTrace(context: NetworkBuilder): Updater = {
    val n = noInputs(context)
    val zeroTrace = Nd4j.zeros(n * (noOutputs + 1))
    (data: LayerData) => data + ("trace" -> zeroTrace)
  }

  def buildForward(context: NetworkBuilder): Updater = ???

  lazy val toJson = Json.obj(
    "type" -> Json.fromString("DENSE"),
    "noOutputs" -> Json.fromInt(noOutputs))
}
