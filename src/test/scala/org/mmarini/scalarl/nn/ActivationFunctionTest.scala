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

import org.scalatest.FunSpec
import org.scalatest.Matchers
import org.scalatest.prop.PropertyChecks

import io.circe.Json
import io.circe.yaml
import io.circe.yaml.syntax.AsYaml
import org.nd4j.linalg.factory.Nd4j
import org.scalatest.PropSpec
import org.scalacheck.Gen

class ActivationFunctionTest extends PropSpec with PropertyChecks with Matchers {
  val Epsilon = 1e-6

  val tanh = TanhActivationFunction

  def valueGen = Gen.choose(-1.0, 1.0)

  property("""Given an Activation Function
  and a initial layer data with 2 random input
  when build a activation updater
  and apply it to initial layer
  then should result the layer with activaetd outputs""") {
    forAll(
      (valueGen, "value")) {
        value =>
          val inputs = Nd4j.ones(2).mul(value)
          val inputData = Map("inputs" -> inputs)
          val updater = tanh.buildActivation
          val newData = updater(inputData)

          val outputs = newData("outputs")

          val x = inputData("inputs").getDouble(0L);
          val y = Nd4j.ones(2).mul(Math.tanh(x));

          newData.get("outputs") should contain(y)

        //          for { i <- 0L until 2 } {
        //            val y = outputs.getDouble(i)
        //            y shouldBe expected +- Epsilon
        //          }
      }
  }

  property("""Given an Activation Function
  and a initial layer data with 2 random input
  when build a gradient updater
  and apply it to initial layer
  then should result the layer with gradient""") {
    forAll(
      (valueGen, "value")) {
        value =>
          val inputs = Nd4j.ones(2).mul(value)
          val y = Math.tanh(value)
          val outputs = Nd4j.ones(2).mul(y)
          val inputData = Map(
            "inputs" -> inputs,
            "outputs" -> outputs)
          val updater: Updater = tanh.buildGradient
          val newData = updater(inputData)

          val g = (1 - y) * (1 + y)
          val expected = Nd4j.ones(2).mul(g)

          newData.get("gradient") should contain(expected)
      }
  }
}
