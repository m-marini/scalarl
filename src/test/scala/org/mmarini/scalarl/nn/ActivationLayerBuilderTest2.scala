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
import org.scalatest.GivenWhenThen
import org.scalatest.Matchers

class ActivationLayerBuilder2Test extends FunSpec with GivenWhenThen with Matchers {
  describe("an ActivationLayerBuilder") {
    it("should create a json doc") {
      Given("an ActivationLayerBuilder")
      val l = ActivationLayerBuilder("l", TanhActivationFunction)

      When("create json")
      val json = l.toJson

      Then("should be an object")
      json.isObject shouldBe true

      And("should have id l")
      json.asObject.flatMap(_("id")).flatMap(_.asString) should contain("l")

      And("should have type ACTIVATION")
      json.asObject.flatMap(_("type")).flatMap(_.asString) should contain("ACTIVATION")

      And("should have activation TANH")
      json.asObject.flatMap(_("activation")).flatMap(_.asString) should contain("TANH")
    }
  }
}
