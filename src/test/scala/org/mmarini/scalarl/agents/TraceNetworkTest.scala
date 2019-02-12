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

package org.mmarini.scalarl.agents

import org.scalatest.Matchers

import scala.math.tanh
import org.scalatest.GivenWhenThen
import org.scalatest.prop.PropertyChecks
import org.scalatest.PropSpec
import org.scalatest.FunSpec
import org.scalacheck.Gen
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.api.ops.impl.transforms.Tanh

class TraceNetworkTest extends PropSpec with PropertyChecks with Matchers {
  val MaxInputs = 5
  val MaxOutputs = 5

  def createNetwork(n: Long, m: Long): TraceNetwork = {
    val layer1 = TraceDenseLayer(n, m, gamma = 0.9, lambda = 0.9, learningRate = 1e-3)
    val layer2 = new TraceTanhLayer()
    new TraceNetwork(Array(layer1, layer2))
  }

  def createInput(noInputs: Long): INDArray = Nd4j.randn(1L, noInputs)

  def createExpected(input: INDArray, nn: TraceNetwork): Array[INDArray] = {
    val layer1 = nn.layers(0).asInstanceOf[TraceDenseLayer]
    val w = layer1.weights
    val b = layer1.bias

    val ws = w.shape()
    val bs = b.shape()

    val act1 = input.mmul(w).addi(b)
    val act2 = Nd4j.getExecutioner().execAndReturn(new Tanh(act1.dup()))
    Array(input, act1, act2)
  }

  property(s"""Given a TraceNetwork
    When forward invoked
    Then should return the expected value""") {
    forAll(
      (Gen.choose(1, MaxInputs), "noInputs"),
      (Gen.choose(1, MaxOutputs), "noOutputs")) {
        (noInputs, noOutputs) =>
          whenever(noInputs >= 1 && noOutputs >= 1) {
            val input = createInput(noInputs)
            val nn = createNetwork(noInputs, noOutputs)

            val output = nn.forward(input)
            val expected = createExpected(input, nn)

            output(0).distance1(expected(0)) should be < 1e-6
            output(1).distance1(expected(1)) should be < 1e-6
            output(2).distance1(expected(2)) should be < 1e-6
          }
      }
  }
}