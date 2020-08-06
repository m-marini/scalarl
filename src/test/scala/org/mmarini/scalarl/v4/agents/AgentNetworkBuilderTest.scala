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

package org.mmarini.scalarl.v4.agents

import org.nd4j.linalg.factory.Nd4j._
import org.scalatest.{FunSpec, Matchers}

class AgentNetworkBuilderTest extends FunSpec with Matchers {

  create()

  describe("AgentNetworkBuilder") {
    it("should validate shortcuts") {
      val shortcuts = Seq(Seq(0, 2), Seq(1, 2))
      val noHidden = 2
      val map = AgentNetworkBuilder.validateShortCut(shortcuts, noHidden).get
      map should contain(2 -> Seq(0, 1))
    }

    it("should compute input layers") {
      val noInputs = 10
      val hiddens = Seq(5, 5)
      val shortcuts = Map(3 -> Seq(0, 1))
      val (inputs, names) = AgentNetworkBuilder.inputLayersByLayer(noInputs, hiddens, shortcuts)
      inputs should contain theSameElementsAs Seq(10, 5, 10 + 5 + 5)
      names should contain theSameElementsAs Seq(Seq("L0"), Seq("L1"), Seq("L2", "L0", "L1"))
    }
  }
}