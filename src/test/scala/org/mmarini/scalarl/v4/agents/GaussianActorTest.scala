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

import org.mmarini.scalarl.v4.Utils._
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j._
import org.nd4j.linalg.ops.transforms.Transforms._
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
  private val denormalize = linearTransf(Range)
  private val normalize = linearInverse(Range)

  def actor(eta: INDArray): GaussianActor = GaussianActor(dimension = 0,
    eta = eta,
    denormalize = denormalize,
    normalize = normalize)

  describe("GaussianActor") {
    it("should compute mu h sigma") {
      val out = ones(2)
      val outs = Array(out, out)
      val a = actor(Eta)

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
        val out = normalize(hstack(mu, h))
        val outs = Array(out, out)

        describe("with delta > 0") {
          val delta = ones(1)

          val (_, _, mu1, h1) = actor(eta).muHStar(outs, actions, delta)

          it("should return mu' = mu") {
            mu1 shouldBe mu
          }
          it("should return sigma' < sigma") {
            h1.getDouble(0L) should be < h.getDouble(0L)
          }
        }

        describe("with delta < 0") {
          val delta = create(Array[Double](-1))
          val (_, _, mu1, h1) = actor(eta).muHStar(outs, actions, delta)

          it("should return mu' = mu") {
            mu1 shouldBe mu
          }
          it("should return sigma' > sigma") {
            h1.getDouble(0L) should be > h.getDouble(0L)
          }
        }
      }

      describe("whith action > mu") {
        val actions = mu.add(1)
        val out = normalize(hstack(mu, h))
        val outs = Array(out, out)
        describe("with delta > 0") {
          val delta = create(Array[Double](1))
          val (_, _, mu1, h1) = actor(eta).muHStar(outs, actions, delta)

          it("should return mu' > mu") {
            mu1.getDouble(0L) should be > mu.getDouble(0L)
          }
          it("should return sigma' < sigma") {
            h1.getDouble(0L) should be < h.getDouble(0L)
          }
        }

        describe("with delta < 0") {
          val delta = create(Array[Double](-1))
          val (_, _, mu1, h1) = actor(eta).muHStar(outs, actions, delta)

          it("should return mu' < mu") {
            mu1.getDouble(0L) should be < mu.getDouble(0L)
          }
          it("should return sigma' > sigma") {
            h1.getDouble(0L) should be > h.getDouble(0L)
          }
        }
      }

      describe("whith action < mu") {
        val actions = mu.sub(1)
        val out = normalize(hstack(mu, h))
        val outs = Array(out, out)
        describe("with delta > 0") {
          val delta = create(Array[Double](1))
          val (_, _, mu1, h1) = actor(eta).muHStar(outs, actions, delta)

          it("should return mu' < mu") {
            mu1.getDouble(0L) should be < mu.getDouble(0L)
          }
          it("should return sigma' < sigma") {
            h1.getDouble(0L) should be < h.getDouble(0L)
          }
        }

        describe("with delta < 0") {
          val delta = create(Array[Double](-1))
          val (_, _, mu1, h1) = actor(eta).muHStar(outs, actions, delta)

          it("should return mu' > mu") {
            mu1.getDouble(0L) should be > mu.getDouble(0L)
          }
          it("should return sigma' > sigma") {
            h1.getDouble(0L) should be > h.getDouble(0L)
          }
        }
      }

      describe("whith action >> mu") {
        val actions = mu.add(4.0)
        val out = normalize(hstack(mu, h))
        val outs = Array(out, out)
        describe("with delta > 0") {
          val delta = create(Array[Double](1))
          val (_, _, mu1, h1) = actor(eta).muHStar(outs, actions, delta)

          it("should return mu' > mu") {
            mu1.getDouble(0L) should be > mu.getDouble(0L)
          }
          it("should return sigma' > sigma") {
            h1.getDouble(0L) should be > h.getDouble(0L)
          }
        }

        describe("with delta < 0") {
          val delta = create(Array[Double](-1))
          val (_, _, mu1, h1) = actor(eta).muHStar(outs, actions, delta)

          it("should return mu' < mu") {
            mu1.getDouble(0L) should be < mu.getDouble(0L)
          }
          it("should return sigma' < sigma") {
            h1.getDouble(0L) should be < h.getDouble(0L)
          }
        }
      }

      describe("whith action << mu") {
        val actions = mu.sub(4.0)
        val out = normalize(hstack(mu, h))
        val outs = Array(out, out)
        describe("with delta > 0") {
          val delta = create(Array[Double](1))
          val (_, _, mu1, h1) = actor(eta).muHStar(outs, actions, delta)

          it("should return mu' < mu") {
            mu1.getDouble(0L) should be < mu.getDouble(0L)
          }
          it("should return sigma' > sigma") {
            h1.getDouble(0L) should be > h.getDouble(0L)
          }
        }

        describe("with delta < 0") {
          val delta = create(Array[Double](-1))
          val (_, _, mu1, h1) = actor(eta).muHStar(outs, actions, delta)

          it("should return mu' > mu") {
            mu1.getDouble(0L) should be > mu.getDouble(0L)
          }
          it("should return sigma' < sigma") {
            h1.getDouble(0L) should be < h.getDouble(0L)
          }
        }
      }
    }
  }
}
