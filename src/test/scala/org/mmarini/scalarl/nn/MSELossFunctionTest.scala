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

class MSELossFunctionTest extends PropSpec with PropertyChecks with Matchers {
  val Epsilon = 1e-6

  val loss = MSELossFunction

  def valueGen = Gen.choose(-1.0, 1.0)

  property("""Given an Activation Function
  and a initial layer data with 2 random input
  when build a activation updater
  and apply it to initial layer
  then should result the layer with activaetd outputs""") {
    forAll(
      (valueGen, "y"),
      (valueGen, "label")) {
        (y, label) =>
          val outputs = Nd4j.ones(2).mul(y)
          val labels = Nd4j.ones(2).mul(label)
          val mask = Nd4j.create(Array(0.0, 1.0))

          val inputData = Map(
            "outputs" -> outputs,
            "labels" -> labels,
            "mask" -> mask)

          val updater = loss.buildGradient
          val newData = updater(inputData)

          val expectedDelta = mask.mul(label - y)

          newData.get("delta") should contain(expectedDelta)
      }
  }
}
