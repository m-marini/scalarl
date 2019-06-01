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

import org.mmarini.scalarl.Feedback
import org.mmarini.scalarl.INDArrayObservation
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import org.scalatest.FunSpec
import org.scalatest.Matchers
import scala.math.abs
import java.io.File

class TraceModelSerializerTest extends FunSpec with Matchers {
  val Outputs = 2
  val Inputs = 3 + Outputs
  val Eps = 1e-3
  val Gamma = 0.9
  val Lambda = 0.8
  val Alpha = 3e-3
  val Filename = "test.zip"

  def createNet(): TraceNetwork = {
    val layer0 = TraceDenseLayer(
      noInputs = Inputs,
      noOutputs = Outputs,
      gamma = Gamma,
      lambda = Lambda,
      learningRate = Alpha,
      traceUpdater = AccumulateTraceUpdater)
    val layer1 = TraceTanhLayer()
    // Creates the output layer
    val outLayer = TraceDenseLayer(
      noInputs = Outputs,
      noOutputs = Outputs,
      gamma = Gamma,
      lambda = Lambda,
      learningRate = Alpha,
      traceUpdater = AccumulateTraceUpdater)
    new TraceNetwork(layers = Array(layer0, layer1, outLayer))
  }

  describe(s"""Given a TraceNetwork""") {
    val net = createNet()
    describe(s"""When TraceModelSerialize.writeModel""") {
      new File(Filename).delete()
      TraceModelSerializer.writeModel(net, Filename)
      it(s"""Then it should create the file""") {
        new File(Filename).exists() shouldBe true
      }
    }
  }

  describe(s"""Given a TraceNetwork
When TraceModelSerialize.writeModel
 and TraceModelSerialize.restoreTraceNetwork""") {
    val net = createNet()
    new File(Filename).delete()
    TraceModelSerializer.writeModel(net, Filename)
    val net1 = TraceModelSerializer.restoreTraceNetwork(new File(Filename))
    it("Then should return 3 layers") {
      net1.layers.length shouldBe 3
    }
    it(" and layer1 should have the same parameters") {
      net1.layers(0) shouldBe a[TraceDenseLayer]

      val layer0 = net.layers(0).asInstanceOf[TraceDenseLayer]
      val layer1 = net1.layers(0).asInstanceOf[TraceDenseLayer]
      layer1.gamma shouldBe (layer0.gamma)
      layer1.lambda shouldBe (layer0.lambda)
      layer1.learningRate shouldBe (layer0.learningRate)
      layer1.weights shouldBe (layer0.weights)
      layer1.bias shouldBe (layer0.bias)
    }
    it(" and layer2 should have the same parameters") {
      net1.layers(1) shouldBe a[TraceTanhLayer]
    }
    it(" and layer3 should have the same parameters") {
      net1.layers(2) shouldBe a[TraceDenseLayer]

      val layer0 = net.layers(2).asInstanceOf[TraceDenseLayer]
      val layer1 = net1.layers(2).asInstanceOf[TraceDenseLayer]
      layer1.gamma shouldBe (layer0.gamma)
      layer1.lambda shouldBe (layer0.lambda)
      layer1.learningRate shouldBe (layer0.learningRate)
      layer1.weights shouldBe (layer0.weights)
      layer1.bias shouldBe (layer0.bias)
    }
  }
}
