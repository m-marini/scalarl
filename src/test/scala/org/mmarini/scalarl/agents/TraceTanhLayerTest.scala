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

class TraceTanhLayerTest extends PropSpec with PropertyChecks with Matchers {
  val MaxInputs = 5
  val MaxOutputs = 5

  def expectedOut(input: INDArray): INDArray = {
    val n = input.size(1)
    val out = Nd4j.zeros(1, n)
    for {
      i <- 0L until n
    } {
      out.putScalar(Array(0, i), tanh(input.getDouble(i)))
    }
    out
  }

  def createInput(noInputs: Long): INDArray = Nd4j.randn(Array(1L, noInputs))

  property(s"""Given a TraceTanhLayer
    When forward invoked
    Then should return the expected value""") {
    forAll(
      (Gen.choose(1, MaxInputs), "noInputs")) {
        (noInputs) =>
          whenever(noInputs >= 1) {
            val input = createInput(noInputs)

            val expected = expectedOut(input)

            val output = new TraceTanhLayer().forward(input)

            output should be(expected)
          }
      }
  }

  private def createInputErrors(output: INDArray, errors: INDArray) = {
    val n = output.size(1)
    val inErrors = Nd4j.zeros(1, n)

    for {
      i <- 0L until n
    } {
      val y = output.getDouble(0L, i)
      val e = errors.getDouble(0L, i)
      inErrors.putScalar(Array(0L, i), (1 - y) * (1 + y) * e)
    }
    inErrors
  }

  property(s"""Given a TraceDenseLayer
      And a single output mask
    When backward invoked
    Then should result the backward error
      And the layer with updated parameters""") {
    forAll(
      (for {
        noOutputs <- Gen.choose(1L, MaxOutputs)
        outputIdx <- Gen.choose(0L, noOutputs - 1)
      } yield (noOutputs, outputIdx), "(noOutputs, outputIdx)")) {
        (outs) =>
          whenever(outs._1 >= 1) {
            outs match {
              case (noOutputs, outputIdx) =>
                val input = createInput(noOutputs)

                val mask = Nd4j.zeros(1L, noOutputs)
                mask.putScalar(Array(outputIdx), 1.0)

                val errors = Nd4j.randn(Array(1L, noOutputs)).muli(mask)
                val layer = new TraceTanhLayer()
                val outputs = layer.forward(input)
                val expected = createInputErrors(outputs, errors)

                val (inputErrors, inputMask) = layer.backward(input, outputs, errors, mask)

                // Checks for input errors
                for {
                  i <- 0L until noOutputs
                } {
                  inputErrors.getDouble(0L, i) should be(expected.getDouble(0L, i) +- 1e-6)
                }

                // Checks for input mask
                for {
                  i <- 0L until noOutputs
                } {
                  if (i == outputIdx) {
                    inputMask.getDouble(0L, i) should be(1.0)
                  } else {
                    inputMask.getDouble(0L, i) should be(0.0)
                  }
                }
            }
          }
      }
  }

  property(s"""Given a TraceDenseLayer
      And full output mask
    When backward invoked
    Then should result the backward error
      And the layer with updated parameters""") {
    forAll(
      (Gen.choose(1L, MaxOutputs), "noOutputs")) {
        (noOutputs) =>
          whenever(noOutputs >= 1) {
            val input = createInput(noOutputs)

            val mask = Nd4j.ones(1L, noOutputs)

            val errors = Nd4j.randn(Array(1, noOutputs)).muli(mask)

            val layer = new TraceTanhLayer()
            val outputs = layer.forward(input)
            val expected = createInputErrors(outputs, errors)

            val (inputErrors, inputMask) = layer.backward(input, outputs, errors, mask)

            // Checks for input errors
            for {
              i <- 0L until noOutputs
            } {
              inputErrors.getDouble(0L, i) should be(expected.getDouble(0L, i) +- 1e-6)
            }

            // Checks for input mask
            for {
              i <- 0L until noOutputs
            } {
              inputMask.getDouble(0L, i) should be(1.0)
            }
          }
      }
  }
}

