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

package org.mmarini.scalarl.v6.agents

import org.mmarini.scalarl.v6.Configuration.jsonFormString
import org.mmarini.scalarl.v6.Utils._
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j._
import org.scalatest.{FunSpec, Matchers}

class GaussianActorTest extends FunSpec with Matchers {
  private val SigmaRange = 2.0
  private val MuRange = 100.0
  private val HRange = Math.log(SigmaRange)

  create()

  private val Eta: INDArray = ones(2)
  private val Range = create(Array(
    Array(-MuRange, -HRange),
    Array(MuRange, HRange)
  ))
  private val norm = clipAndNormalize(Range)

  private def actor = GaussianActor(dimension = 0,
    alphaMu = 1,
    alphaSigma = 1,
    muRange = create(Array(-MuRange, MuRange)).transposei(),
    sigmaRange = create(Array(1 / SigmaRange, SigmaRange)).transposei())

  describe("GaussianActor") {
    it("should compute mu h sigma") {
      val out = ones(2)
      val outs = Array(out, out)
      val a = actor

      val (mu, h, sigma) = a.muHSigma(outs)

      mu shouldBe ones(1).muli(MuRange)
      h shouldBe ones(1).muli(HRange)
      sigma shouldBe ones(1).muli(SigmaRange)
    }

    describe("compute mu', h'") {
      val mu = create(Array[Double](1))
      val h = create(Array[Double](Math.log(2)))
      val eta = ones(2)

      describe("with action = mu") {
        val actions = mu
        val out = norm(hstack(mu, h))
        val outs = Array(out, out)

        describe("with delta > 0") {
          val delta = ones(1)

          val map = actor.computeLabels(outs, actions, delta)

          it("should return mu* = mu") {
            map("mu*(0)") shouldBe map("mu(0)")
          }
          it("should return h* < h") {
            map("h*(0)").asInstanceOf[INDArray].getDouble(0L) should be < map("h(0)").asInstanceOf[INDArray].getDouble(0L)
          }
        }

        describe("with delta < 0") {
          val delta = create(Array[Double](-1))
          val map = actor.computeLabels(outs, actions, delta)

          it("should return mu* = mu") {
            map("mu*(0)") shouldBe map("mu(0)")
          }
          it("should return h* > h") {
            map("h*(0)").asInstanceOf[INDArray].getDouble(0L) should be > map("h(0)").asInstanceOf[INDArray].getDouble(0L)
          }
        }
      }

      describe("with action > mu") {
        val actions = mu.add(1)
        val out = norm(hstack(mu, h))
        val outs = Array(out, out)
        describe("with delta > 0") {
          val delta = create(Array[Double](1))
          val map = actor.computeLabels(outs, actions, delta)

          it("should return mu* > mu") {
            map("mu*(0)").asInstanceOf[INDArray].getDouble(0L) should be > map("mu(0)").asInstanceOf[INDArray].getDouble(0L)
          }
          it("should return h* < h") {
            map("h*(0)").asInstanceOf[INDArray].getDouble(0L) should be < map("h(0)").asInstanceOf[INDArray].getDouble(0L)
          }
        }

        describe("with delta < 0") {
          val delta = create(Array[Double](-1))
          val map = actor.computeLabels(outs, actions, delta)

          it("should return mu* < mu") {
            map("mu*(0)").asInstanceOf[INDArray].getDouble(0L) should be < map("mu(0)").asInstanceOf[INDArray].getDouble(0L)
          }
          it("should return h* > h") {
            map("h*(0)").asInstanceOf[INDArray].getDouble(0L) should be > map("h(0)").asInstanceOf[INDArray].getDouble(0L)
          }
        }
      }

      describe("with action < mu") {
        val actions = mu.sub(1)
        val out = norm(hstack(mu, h))
        val outs = Array(out, out)
        describe("with delta > 0") {
          val delta = create(Array[Double](1))
          val map = actor.computeLabels(outs, actions, delta)

          it("should return mu* < mu") {
            map("mu*(0)").asInstanceOf[INDArray].getDouble(0L) should be < map("mu(0)").asInstanceOf[INDArray].getDouble(0L)
          }
          it("should return h* < h") {
            map("h*(0)").asInstanceOf[INDArray].getDouble(0L) should be < map("h(0)").asInstanceOf[INDArray].getDouble(0L)
          }
        }

        describe("with delta < 0") {
          val delta = create(Array[Double](-1))
          val map = actor.computeLabels(outs, actions, delta)

          it("should return mu* > mu") {
            map("mu*(0)").asInstanceOf[INDArray].getDouble(0L) should be > map("mu(0)").asInstanceOf[INDArray].getDouble(0L)
          }
          it("should return h* > h") {
            map("mu*(0)").asInstanceOf[INDArray].getDouble(0L) should be > map("mu(0)").asInstanceOf[INDArray].getDouble(0L)
          }
        }
      }

      describe("with action >> mu") {
        val actions = mu.add(4.0)
        val out = norm(hstack(mu, h))
        val outs = Array(out, out)
        describe("with delta > 0") {
          val delta = create(Array[Double](1))
          val map = actor.computeLabels(outs, actions, delta)

          it("should return mu* > mu") {
            map("mu*(0)").asInstanceOf[INDArray].getDouble(0L) should be > map("mu(0)").asInstanceOf[INDArray].getDouble(0L)
          }
          it("should return h* > h") {
            map("h*(0)").asInstanceOf[INDArray].getDouble(0L) should be > map("h(0)").asInstanceOf[INDArray].getDouble(0L)
          }
        }

        describe("with delta < 0") {
          val delta = create(Array[Double](-1))
          val map = actor.computeLabels(outs, actions, delta)

          it("should return mu* < mu") {
            map("mu*(0)").asInstanceOf[INDArray].getDouble(0L) should be < map("mu(0)").asInstanceOf[INDArray].getDouble(0L)
          }
          it("should return h* < h") {
            map("h*(0)").asInstanceOf[INDArray].getDouble(0L) should be < map("h(0)").asInstanceOf[INDArray].getDouble(0L)
          }
        }
      }

      describe("with action << mu") {
        val actions = mu.sub(4.0)
        val out = norm(hstack(mu, h))
        val outs = Array(out, out)
        describe("with delta > 0") {
          val delta = create(Array[Double](1))
          val map = actor.computeLabels(outs, actions, delta)

          it("should return mu* < mu") {
            map("mu*(0)").asInstanceOf[INDArray].getDouble(0L) should be < map("mu(0)").asInstanceOf[INDArray].getDouble(0L)
          }
          it("should return h* > h") {
            map("h*(0)").asInstanceOf[INDArray].getDouble(0L) should be > map("h(0)").asInstanceOf[INDArray].getDouble(0L)
          }
        }

        describe("with delta < 0") {
          val delta = create(Array[Double](-1))
          val map = actor.computeLabels(outs, actions, delta)

          it("should return mu* > mu") {
            map("mu*(0)").asInstanceOf[INDArray].getDouble(0L) should be > map("mu(0)").asInstanceOf[INDArray].getDouble(0L)
          }
          it("should return h* < h") {
            map("h*(0)").asInstanceOf[INDArray].getDouble(0L) should be < map("h(0)").asInstanceOf[INDArray].getDouble(0L)
          }
        }
      }
    }
  }
  describe("create from json") {
    val conf = jsonFormString(
      """
        |---
        |actor:
        |    alphaMu: 0.1
        |    alphaSigma: 0.2
        |    muRange:
        |    - [-2, 2]
        |    sigmaRange:
        |    - [0.01, 0.1]
        |""".stripMargin)

    val actor = GaussianActor.fromJson(conf.hcursor.downField("actor"))(0).get

    actor.eta shouldBe create(Array(0.1, 0.2))

    actor.denormalize(ones(2).muli(-1.1)) shouldBe create(Array(-2, Math.log(0.01)))
    actor.denormalize(ones(2).negi()) shouldBe create(Array(-2, Math.log(0.01)))
    actor.denormalize(zeros(2)) shouldBe create(Array(0.0, (Math.log(0.01) + Math.log(0.1)) / 2))
    actor.denormalize(ones(2)) shouldBe create(Array(2.0, Math.log(0.1)))
    actor.denormalize(ones(2).muli(1.1)) shouldBe create(Array(2.0, Math.log(0.1)))

    actor.normalize(create(Array(-2.1, Math.log(0.009)))) shouldBe ones(2).negi()
    actor.normalize(create(Array(-2.0, Math.log(0.01)))) shouldBe ones(2).negi()
    actor.normalize(create(Array(0.0, (Math.log(0.01) + Math.log(0.1)) / 2))) shouldBe zeros(2)
    actor.normalize(create(Array(2.0, Math.log(0.1)))) shouldBe ones(2)
    actor.normalize(create(Array(2.1, Math.log(0.11)))) shouldBe ones(2)
  }
}
