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

import org.nd4j.linalg.factory.Nd4j
import org.scalacheck.Gen
import org.scalatest.prop.PropertyChecks
import org.scalatest.{Matchers, PropSpec}

class ActivationLayerBuilderTest extends PropSpec with PropertyChecks with Matchers {
  val Epsilon = 1e-6

  val tanhLayer = ActivationLayerBuilder("tanh", TanhActivationFunction)

  def valueGen = Gen.choose(-1.0, 1.0)

  def inputsDataGen = inputsGen.map(inputs => Map("tanh.inputs" -> inputs))

  def inputsGen = Gen.choose(-1.0, 1.0).map(v => Nd4j.ones(2).mul(v))

  private def buildInputs(value: Double) = {
    val inputs = Nd4j.ones(2).mul(value)
    val data = Map("tanh.inputs" -> inputs)
    (inputs, data)
  }

  property(
    """
Given an Activation layer builder
  and a initial layer data with 2 random input
  when build a forward updater
  and apply to initial layer
  then should result the layer with activated outputs""") {
    forAll(
      (valueGen, "value")) {
      value =>
        val (inputs, inputsData) = buildInputs(value)
        val updater = tanhLayer.forwardBuilder(None.orNull)
        val newData = updater.build(inputsData)

        val y = Math.tanh(value)
        val expected = Nd4j.ones(2).mul(y)
        newData.get("tanh.outputs") should contain(expected)
    }
  }

  property(
    """
Given an Activation layer builder
  and a initial layer data with 2 random input
  when build a gradient updater
  and apply to initial layer
  then should result the layer with gradient""") {
    forAll(
      (valueGen, "value")) {
      value =>
        val (inputs, inputsData) = buildInputs(value)
        val updater = tanhLayer.gradientBuilder(None.orNull).build
        val y = Math.tanh(value)
        val withOutputs = inputsData + ("tanh.outputs" -> Nd4j.ones(2).mul(y))
        val newData = updater(withOutputs)

        newData should be theSameInstanceAs (withOutputs)
    }
  }

  property(
    """
Given an Activation layer builder
  and a initial layer data with 2 random input and delta and gradient
  when build a delta updater
  and apply to initial layer
  then should result the delta input""") {
    forAll(
      (valueGen, "value"),
      (valueGen, "delta")) {
      (value, delta) =>
        val (inputs, inputsData) = buildInputs(value)
        val grad = 1 - value * value
        val expectedDelta = delta * grad

        val withDelta = inputsData +
          ("tanh.delta" -> Nd4j.ones(2).mul(delta)) +
          ("tanh.outputs" -> Nd4j.ones(2).mul(value))

        val updater = tanhLayer.deltaBuilder(None.orNull).build
        val newData = updater(withDelta)

        val expected = Nd4j.ones(2).mul(expectedDelta)
        newData.get("tanh.inputDelta") should contain(expected)
    }
  }
}
