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
import org.scalatest.GivenWhenThen
import org.scalatest.prop.PropertyChecks
import org.scalatest.PropSpec
import org.scalatest.FunSpec
import org.scalacheck.Gen
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.api.ndarray.INDArray

class TraceDenseLayerTest extends PropSpec with PropertyChecks with Matchers {
  val MaxSamples = 10
  val MaxInputs = 5
  val MaxOutputs = 5

  def expectedOut(input: INDArray, weights: INDArray, bias: INDArray): INDArray = {
    val n = input.size(0)
    val ni = input.size(1)
    val no = weights.size(1)
    val out = Nd4j.zeros(Array(n, no), 'c')
    for {
      i <- 0L until n
      j <- 0L until no
    } {
      var x = 0.0
      for { k <- 0L until ni } {
        x = x + input.getDouble(i, k) * weights.getDouble(k, j)
      }
      out.putScalar(Array(i, j), x + bias.getDouble(j))
    }
    out
  }

  def createInput(noSamples: Int, noInputs: Int): INDArray = Nd4j.randn('c', Array(noSamples, noInputs))

  def createWeights(noInputs: Int, noOutputs: Int): INDArray = Nd4j.randn('c', Array(noInputs, noOutputs))

  def createBias(noOutputs: Int): INDArray = Nd4j.randn('c', Array(noOutputs))

  property(s""""Given a TraceDenseLayer
    When forward invoked
    Then should return the expected value""") {
    forAll(
      (Gen.choose(1, MaxSamples), "noSamples"),
      (Gen.choose(1, MaxInputs), "noInputs"),
      (Gen.choose(1, MaxOutputs), "noOutputs")) {
        (noSamples, noInputs, noOutputs) =>
          whenever(noSamples >= 1 && noInputs >= 1 && noOutputs >= 1) {
            val input = createInput(noSamples, noInputs)
            val weights = createWeights(noInputs, noOutputs);
            val bias = createBias(noOutputs)

            val noParms = weights.length() + bias.length()

            val trace = Nd4j.zeros(Array(noParms), 'c')
            val expected = expectedOut(input, weights, bias)

            val layer = new TraceDenseLayer(weights, bias, trace)
            val output = layer.forward(input)

            output should be(expected)
          }
      }
  }

  property(s""""Given a TraceDenseLayer
    When gradien invoked
    Then should return the expected value""") {
    forAll(
      (Gen.choose(1, MaxSamples), "noSamples"),
      (Gen.choose(1, MaxInputs), "noInputs"),
      (Gen.choose(1, MaxOutputs), "noOutputs")) {
        (noSamples, noInputs, noOutputs) =>
          whenever(noSamples >= 1 && noInputs >= 1 && noOutputs >= 1) {
            val input = createInput(noSamples, noInputs)
            val weights = createWeights(noInputs, noOutputs);
            val bias = createBias(noOutputs)
            val noParms = weights.length() + bias.length()
            val trace = Nd4j.zeros(Array(noParms), 'c')

            val layer = new TraceDenseLayer(weights, bias, trace)
            val outputs = layer.forward(input)
            val (gradW, gradB) = layer.gradient((input, outputs))

            gradB.rank() should be(2)
            gradB.shape() should be(Array(noSamples, noOutputs))
            for {
              i <- 0 until noSamples
              j <- 0 until noOutputs
            } {
              gradB.getDouble(i.toLong, j.toLong) should be(1.0)
            }

            gradW.rank() should be(3)
            gradW.shape() should be(Array(noSamples, noInputs, noOutputs))
            for {
              i <- 0 until noSamples
              j <- 0 until noInputs
              k <- 0 until noOutputs
            } {
              gradW.getDouble(i.toLong, j.toLong, k.toLong) should be(input.getDouble(i.toLong, j.toLong))
            }

          }
      }
  }
}
