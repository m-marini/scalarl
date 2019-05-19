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
  val MaxInputs = 5
  val MaxOutputs = 5
  val Epsilon = 1e-3

  def expectedOut(input: INDArray, weights: INDArray, bias: INDArray): INDArray = {
    val ni = input.size(1)
    val no = weights.size(1)
    val out = Nd4j.zeros(1, no)
    for {
      i <- 0L until no
    } {
      var x = 0.0
      for { k <- 0L until ni } {
        x = x + input.getDouble(k) * weights.getDouble(k, i)
      }
      out.putScalar(Array(0, i), x + bias.getDouble(i))
    }
    out
  }

  def createInput(noInputs: Long): INDArray = Nd4j.randn(Array(1L, noInputs))

  def createWeights(noInputs: Long, noOutputs: Long): INDArray = Nd4j.randn(Array(noInputs, noOutputs))

  def createBias(noOutputs: Long): INDArray = Nd4j.randn(Array(1L, noOutputs))

  property(s"""Given a TraceDenseLayer
    When forward invoked
    Then should return the expected value""") {
    forAll(
      (Gen.choose(1, MaxInputs), "noInputs"),
      (Gen.choose(1, MaxOutputs), "noOutputs"),
      (Gen.choose(0.0, 1.0), "gamma"),
      (Gen.choose(0.0, 1.0), "lambda")) {
        (noInputs, noOutputs, gamma, lambda) =>
          whenever(noInputs >= 1 && noOutputs >= 1) {
            val input = createInput(noInputs)
            val weights = createWeights(noInputs, noOutputs);
            val bias = createBias(noOutputs)

            val expected = expectedOut(input, weights, bias)

            val layer = TraceDenseLayer(weights, bias, gamma, lambda, Epsilon)
            val output = layer.forward(input)

            output should be(expected)
          }
      }
  }

  property(s"""Given a TraceDenseLayer
    When gradien invoked
    Then should return the expected value""") {
    forAll(
      (Gen.choose(1, MaxInputs), "noInputs"),
      (Gen.choose(1, MaxOutputs), "noOutputs"),
      (Gen.choose(0.0, 1.0), "gamma"),
      (Gen.choose(0.0, 1.0), "lambda")) {
        (noInputs, noOutputs, gamma, lambda) =>
          whenever(noInputs >= 1 && noOutputs >= 1) {
            val input = createInput(noInputs)
            val weights = createWeights(noInputs, noOutputs);
            val bias = createBias(noOutputs)

            val layer = TraceDenseLayer(weights, bias, gamma, lambda, Epsilon)
            val outputs = layer.forward(input)
            val (gradW, gradB) = layer.gradient(input, outputs)

            gradB.rank() should be(2)
            gradB.shape() should be(Array(1, noOutputs))
            for {
              i <- 0L until noOutputs
            } {
              gradB.getDouble(i) should be(1.0)
            }

            gradW.rank() should be(2)
            gradW.shape() should be(Array(noInputs, noOutputs))
            for {
              i <- 0L until noInputs
              j <- 0 until noOutputs
            } {
              gradW.getDouble(i.toLong, j.toLong) should be(input.getDouble(i.toLong))
            }

          }
      }
  }

  private def createInputErrors(weights: INDArray, errors: INDArray) = {
    val Array(ni, no) = weights.shape()
    val inErrors = Nd4j.zeros(1, ni)

    for {
      i <- 0L until ni
    } {
      val x = for {
        j <- 0L until no
      } yield weights.getDouble(i, j) * errors.getDouble(0L, j)
      inErrors.putScalar(Array(0L, i), x.sum)
    }
    inErrors
  }

  property(s"""Given a TraceDenseLayer
      And a single output mask
    When backward invoked
    Then should result the backward error
      And the layer with updated parameters""") {
    forAll(
      (Gen.choose(1, MaxInputs), "noInputs"),
      (for {
        noOutputs <- Gen.choose(1, MaxOutputs)
        outputIdx <- Gen.choose(0, noOutputs - 1)
      } yield (noOutputs, outputIdx), "(noOutputs, outputIdx)"),
      (Gen.choose(0.0, 1.0), "gamma"),
      (Gen.choose(0.0, 1.0), "lambda"),
      (Gen.choose(0.0, 1.0), "learningRate")) {
        (noInputs, outs, gamma, lambda, learningRate) =>
          whenever(noInputs >= 1 && outs._1 >= 1) {
            outs match {
              case (noOutputs, outputIdx) =>
                val input = createInput(noInputs)
                val initWeights = createWeights(noInputs, noOutputs)
                val initBias = createBias(noOutputs)

                val mask = Nd4j.zeros(noOutputs)
                mask.putScalar(Array(0L, outputIdx.toLong), 1.0)

                val errors = Nd4j.randn(Array(1, noOutputs)).muli(mask)

                val expected = createInputErrors(initWeights, errors)

                val layer = TraceDenseLayer(initWeights.dup(), initBias.dup(), gamma, lambda, learningRate)
                val outputs = layer.forward(input)
                val (layer1, inputErrors, inputMask) = layer.backward(input, outputs, errors, mask)

                // Checks for input errors
                for {
                  i <- 0L until noInputs
                } {
                  inputErrors.getDouble(0L, i) should be(expected.getDouble(0L, i) +- 1e-6)
                }

                // Checks for input mask
                for {
                  i <- 0L until noInputs
                } {
                  inputMask.getDouble(0L, i) should be(1.0)
                }

                // Checks for weights
                val weights = layer1.asInstanceOf[TraceDenseLayer].weights
                for {
                  i <- 0L until noInputs
                  j <- 0L until noOutputs
                } {
                  if (j == outputIdx) {
                    weights.getDouble(i, j) should be(
                      initWeights.getDouble(i, j) +
                        input.getDouble(0L, i) * learningRate * errors.getDouble(0L, j) +- 1e-6)
                  } else {
                    weights.getDouble(i, j) should be(initWeights.getDouble(i, j))
                  }
                }

                // Checks for bias
                val bias = layer1.asInstanceOf[TraceDenseLayer].bias
                for {
                  j <- 0L until noOutputs
                } {
                  if (j == outputIdx) {
                    bias.getDouble(0L, j) should be(
                      initBias.getDouble(0L, j) +
                        learningRate * errors.getDouble(0L, j) +- 1e-6)
                  } else {
                    bias.getDouble(0L, j) should be(initBias.getDouble(0L, j))
                  }
                }

                // Checks for weight traces
                val wTraces = layer1.asInstanceOf[TraceDenseLayer].weightTraces
                for {
                  i <- 0L until noInputs
                  j <- 0L until noOutputs
                } {
                  if (j == outputIdx) {
                    wTraces.getDouble(i, j) should be(input.getDouble(0L, i))
                  } else {
                    wTraces.getDouble(i, j) should be(0.0)
                  }
                }

                // Checks for bias traces
                val bTraces = layer1.asInstanceOf[TraceDenseLayer].biasTraces
                for {
                  i <- 0L until noOutputs
                } {
                  if (i == outputIdx) {
                    bTraces.getDouble(i) should be(1.0)
                  } else {
                    bTraces.getDouble(i) should be(0.0)
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
      (Gen.choose(1L, MaxInputs), "noInputs"),
      (Gen.choose(1L, MaxOutputs), "noOutputs"),
      (Gen.choose(0.0, 1.0), "gamma"),
      (Gen.choose(0.0, 1.0), "lambda"),
      (Gen.choose(0.0, 1.0), "learningRate")) {
        (noInputs, noOutputs, gamma, lambda, learningRate) =>
          whenever(noInputs >= 1 && noOutputs >= 1) {
            val input = createInput(noInputs)
            val initWeights = createWeights(noInputs, noOutputs)
            val initBias = createBias(noOutputs)

            val mask = Nd4j.ones(1L, noOutputs)

            val errors = Nd4j.randn(Array(1, noOutputs)).muli(mask)

            val expected = createInputErrors(initWeights, errors)

            val layer = TraceDenseLayer(initWeights.dup(), initBias.dup(), gamma, lambda, learningRate)
            val outputs = layer.forward(input)
            val (layer1, inputErrors, inputMask) = layer.backward(input, outputs, errors, mask)

            // Checks for input errors
            for {
              i <- 0L until noInputs
            } {
              inputErrors.getDouble(0L, i) should be(expected.getDouble(0L, i) +- 1e-6)
            }

            // Checks for input mask
            for {
              i <- 0L until noInputs
            } {
              inputMask.getDouble(0L, i) should be(1.0)
            }

            // Checks for weights
            val weights = layer1.asInstanceOf[TraceDenseLayer].weights
            for {
              i <- 0L until noInputs
              j <- 0L until noOutputs
            } {
              weights.getDouble(i, j) should be(
                initWeights.getDouble(i, j) +
                  input.getDouble(0L, i) * learningRate * errors.getDouble(0L, j) +- 1e-6)
            }

            // Checks for bias
            val bias = layer1.asInstanceOf[TraceDenseLayer].bias
            for {
              j <- 0L until noOutputs
            } {
              bias.getDouble(0L, j) should be(
                initBias.getDouble(0L, j) +
                  learningRate * errors.getDouble(0L, j) +- 1e-6)
            }

            // Checks for weight traces
            val wTraces = layer1.asInstanceOf[TraceDenseLayer].weightTraces
            for {
              i <- 0L until noInputs
              j <- 0L until noOutputs
            } {
              wTraces.getDouble(i, j) should be(input.getDouble(0L, i))
            }

            // Checks for bias traces
            val bTraces = layer1.asInstanceOf[TraceDenseLayer].biasTraces
            for {
              i <- 0L until noOutputs
            } {
              bTraces.getDouble(i) should be(1.0)
            }
          }
      }
  }

  property(s"""Given a TraceDenseLayer
      And full output mask
    When backward invoked
      and clear trace invoked
    Then should result the backward error
      And the layer with updated parameters
      And traces resetted""") {
    forAll(
      (Gen.choose(1L, MaxInputs), "noInputs"),
      (Gen.choose(1L, MaxOutputs), "noOutputs"),
      (Gen.choose(0.0, 1.0), "gamma"),
      (Gen.choose(0.0, 1.0), "lambda"),
      (Gen.choose(0.0, 1.0), "learningRate")) {
        (noInputs, noOutputs, gamma, lambda, learningRate) =>
          whenever(noInputs >= 1 && noOutputs >= 1) {
            val input = createInput(noInputs)
            val initWeights = createWeights(noInputs, noOutputs)
            val initBias = createBias(noOutputs)

            val mask = Nd4j.ones(1L, noOutputs)

            val errors = Nd4j.randn(Array(1, noOutputs)).muli(mask)

            val expected = createInputErrors(initWeights, errors)

            val layer = TraceDenseLayer(initWeights.dup(), initBias.dup(), gamma, lambda, learningRate)
            val outputs = layer.forward(input)
            val (layer1, inputErrors, inputMask) = layer.backward(input, outputs, errors, mask)
            val layer2 = layer1.clearTraces()

            // Checks for input errors
            for {
              i <- 0L until noInputs
            } {
              inputErrors.getDouble(0L, i) should be(expected.getDouble(0L, i) +- 1e-6)
            }

            // Checks for input mask
            for {
              i <- 0L until noInputs
            } {
              inputMask.getDouble(0L, i) should be(1.0)
            }

            // Checks for weights
            val weights = layer2.asInstanceOf[TraceDenseLayer].weights
            for {
              i <- 0L until noInputs
              j <- 0L until noOutputs
            } {
              weights.getDouble(i, j) should be(
                initWeights.getDouble(i, j) +
                  input.getDouble(0L, i) * learningRate * errors.getDouble(0L, j) +- 1e-6)
            }

            // Checks for bias
            val bias = layer2.asInstanceOf[TraceDenseLayer].bias
            for {
              j <- 0L until noOutputs
            } {
              bias.getDouble(0L, j) should be(
                initBias.getDouble(0L, j) +
                  learningRate * errors.getDouble(0L, j) +- 1e-6)
            }

            // Checks for weight traces
            val wTraces = layer2.asInstanceOf[TraceDenseLayer].weightTraces
            for {
              i <- 0L until noInputs
              j <- 0L until noOutputs
            } {
              wTraces.getDouble(i, j) should be(0.0)
            }

            // Checks for bias traces
            val bTraces = layer2.asInstanceOf[TraceDenseLayer].biasTraces
            for {
              i <- 0L until noOutputs
            } {
              bTraces.getDouble(i) should be(0.0)
            }
          }
      }
  }
}

