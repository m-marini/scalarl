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

package org.mmarini.scalarl.v6.envs

import com.typesafe.scalalogging.LazyLogging
import org.mmarini.scalarl.v6.agents.Tiles
import org.nd4j.linalg.factory.Nd4j._
import org.nd4j.linalg.ops.transforms.Transforms
import org.scalatest.{FunSpec, Matchers}

class TilesTest extends FunSpec with Matchers with LazyLogging {

  create()

  describe("1D Tiles with 1 tile") {
    val tiles = Tiles(1)
    it("should have 4 tilings") {
      tiles.tilings shouldBe 4

      val a = ones(3).mul(-2)
      val b = ones(3).mul(10)
      val c = create(Array(-2.0, 1, 10))
      val limit = create(Array(2.0, 3, 4))
      Transforms.max(a, 0) shouldBe zeros(3)
      Transforms.min(b, limit) shouldBe limit
      Transforms.min(c, limit) shouldBe create(Array(-2.0, 1, 4))
      Transforms.max(c, 0) shouldBe create(Array(0.0, 1, 10))
    }

    it("should have offset 0,1 2,3") {
      tiles.offsets shouldBe create(Array(0.0, 0.25, 0.5, 0.75)).transpose()
    }

    it("should have 8 features") {
      tiles.noFeatures shouldBe 8
    }

    it("should generate features") {
      tiles.features(create(Array(0.0))) should contain theSameElementsAs Seq(0L, 2, 4, 6)
      tiles.features(create(Array(0.24))) should contain theSameElementsAs Seq(0L, 2, 4, 6)
      tiles.features(create(Array(0.25))) should contain theSameElementsAs Seq(0L, 2, 4, 7)
      tiles.features(create(Array(0.5))) should contain theSameElementsAs Seq(0L, 2, 5, 7)
      tiles.features(create(Array(0.75))) should contain theSameElementsAs Seq(0L, 3, 5, 7)
      tiles.features(create(Array(0.99))) should contain theSameElementsAs Seq(0L, 3, 5, 7)
    }
  }

  describe("1D Tiles with 2 tile") {
    val tiles = Tiles(2)
    it("should have 4 tilings") {
      tiles.tilings shouldBe 4
    }

    it("should have offset 0,1 2,3") {
      tiles.offsets shouldBe create(Array(0.0, 0.25, 0.5, 0.75)).transpose()
    }

    it("should have 12 features") {
      tiles.noFeatures shouldBe 12
    }

    it("should generate features") {
      tiles.features(create(Array(0.0))) should contain theSameElementsAs Seq(0L, 3, 6, 9)
      tiles.features(create(Array(0.24))) should contain theSameElementsAs Seq(0L, 3, 6, 9)
      tiles.features(create(Array(0.25))) should contain theSameElementsAs Seq(0L, 3, 6, 10)
      tiles.features(create(Array(0.5))) should contain theSameElementsAs Seq(0L, 3, 7, 10)
      tiles.features(create(Array(0.75))) should contain theSameElementsAs Seq(0L, 4, 7, 10)
      tiles.features(create(Array(1.00))) should contain theSameElementsAs Seq(1L, 4, 7, 10)
      tiles.features(create(Array(1.25))) should contain theSameElementsAs Seq(1L, 4, 7, 11)
      tiles.features(create(Array(1.5))) should contain theSameElementsAs Seq(1L, 4, 8, 11)
      tiles.features(create(Array(1.75))) should contain theSameElementsAs Seq(1L, 5, 8, 11)
    }
  }

  describe("2D Tiles with 1 tile") {
    val tiles = Tiles(1, 1)
    it("should have 8 tilings") {
      tiles.tilings shouldBe 8
    }

    it("should have 2D offsets") {
      tiles.offsets.mul(8) shouldBe create(Array(
        Array(0.0, 0.0),
        Array(1.0, 3.0),
        Array(2.0, 6.0),
        Array(3.0, 1.0),
        Array(4.0, 4.0),
        Array(5.0, 7.0),
        Array(6.0, 2.0),
        Array(7.0, 5.0)
      ))
    }

    it("should have 32 features") {
      tiles.noFeatures shouldBe 32
    }

    it("should generate x features") {
      tiles.features(create(Array(0.0, 0.0))) should contain theSameElementsAs Seq(0L, 4, 8, 12, 16, 20, 24, 28)
      tiles.features(create(Array(0.125, 0.0))) should contain theSameElementsAs Seq(0L, 4, 8, 12, 16, 20, 24, 29)
      tiles.features(create(Array(0.25, 0.0))) should contain theSameElementsAs Seq(0L, 4, 8, 12, 16, 20, 25, 29)
      tiles.features(create(Array(0.375, 0.0))) should contain theSameElementsAs Seq(0L, 4, 8, 12, 16, 21, 25, 29)
      tiles.features(create(Array(0.5, 0.0))) should contain theSameElementsAs Seq(0L, 4, 8, 12, 17, 21, 25, 29)
      tiles.features(create(Array(0.625, 0.0))) should contain theSameElementsAs Seq(0L, 4, 8, 13, 17, 21, 25, 29)
      tiles.features(create(Array(0.75, 0.0))) should contain theSameElementsAs Seq(0L, 4, 9, 13, 17, 21, 25, 29)
      tiles.features(create(Array(0.875, 0.0))) should contain theSameElementsAs Seq(0L, 5, 9, 13, 17, 21, 25, 29)
    }

    it("should generate y features") {
      tiles.features(create(Array(0.0, 0.0))) should contain theSameElementsAs Seq(0L, 4, 8, 12, 16, 20, 24, 28)
      tiles.features(create(Array(0.0, 0.125))) should contain theSameElementsAs Seq(0L, 4, 8, 12, 16, 22, 24, 28)
      tiles.features(create(Array(0.0, 0.25))) should contain theSameElementsAs Seq(0L, 4, 10, 12, 16, 22, 24, 28)
      tiles.features(create(Array(0.0, 0.375))) should contain theSameElementsAs Seq(0L, 4, 10, 12, 16, 22, 24, 30)
      tiles.features(create(Array(0.0, 0.5))) should contain theSameElementsAs Seq(0L, 4, 10, 12, 18, 22, 24, 30)
      tiles.features(create(Array(0.0, 0.625))) should contain theSameElementsAs Seq(0L, 6, 10, 12, 18, 22, 24, 30)
      tiles.features(create(Array(0.0, 0.75))) should contain theSameElementsAs Seq(0L, 6, 10, 12, 18, 22, 26, 30)
      tiles.features(create(Array(0.0, 0.875))) should contain theSameElementsAs Seq(0L, 6, 10, 14, 18, 22, 26, 30)
    }

    it("should generate out of range features") {
      tiles.features(create(Array(-2.0, -2.0))) should contain theSameElementsAs Seq(0L, 4, 8, 12, 16, 20, 24, 28)
      tiles.features(create(Array(-2.0, 2.0))) should contain theSameElementsAs Seq(2L, 6, 10, 14, 18, 22, 26, 30)
      tiles.features(create(Array(2.0, -2.0))) should contain theSameElementsAs Seq(1L, 5, 9, 13, 17, 21, 25, 29)
      tiles.features(create(Array(2.0, 2.0))) should contain theSameElementsAs Seq(3L, 7, 11, 15, 19, 23, 27, 31)
    }
  }

  describe("2D Tiles with 4 tile") {
    val tiles = Tiles(2, 2)
    it("should have 8 tilings") {
      tiles.tilings shouldBe 8
    }

    it("should have 2D offsets") {
      tiles.offsets.mul(8) shouldBe create(Array(
        Array(0.0, 0.0),
        Array(1.0, 3.0),
        Array(2.0, 6.0),
        Array(3.0, 1.0),
        Array(4.0, 4.0),
        Array(5.0, 7.0),
        Array(6.0, 2.0),
        Array(7.0, 5.0)
      ))
    }

    it("should have 72 features") {
      tiles.noFeatures shouldBe 72
    }

    it("should generate x features") {
      tiles.features(create(Array(0.0, 0.0))) should contain theSameElementsAs Seq(0L, 9, 18, 27, 36, 45, 54, 63)
      tiles.features(create(Array(0.125, 0.0))) should contain theSameElementsAs Seq(0L, 9, 18, 27, 36, 45, 54, 64)
    }

    it("should generate y features") {
      tiles.features(create(Array(0.0, 0.125))) should contain theSameElementsAs Seq(0L, 9, 18, 27, 36, 48, 54, 63)
    }

    it("should generate 1.1 features") {
      tiles.features(create(Array(1.0, 0.0))) should contain theSameElementsAs Seq(1L, 10, 19, 28, 37, 46, 55, 64)
      tiles.features(create(Array(0.0, 1.0))) should contain theSameElementsAs Seq(3L, 12, 21, 30, 39, 48, 57, 66)
      tiles.features(create(Array(1.0, 1.0))) should contain theSameElementsAs Seq(4L, 13, 22, 31, 40, 49, 58, 67)
      tiles.features(create(Array(1.9, 1.9))) should contain theSameElementsAs Seq(4L, 17, 26, 35, 44, 53, 62, 71)
    }
  }

  describe("2D Tiles with 4 tile and hash 10") {
    val tiles = Tiles.withHash(10, 2, 2)
    it("should have 8 tilings") {
      tiles.tilings shouldBe 8
    }

    it("should have 2D offsets") {
      tiles.offsets.mul(8) shouldBe create(Array(
        Array(0.0, 0.0),
        Array(1.0, 3.0),
        Array(2.0, 6.0),
        Array(3.0, 1.0),
        Array(4.0, 4.0),
        Array(5.0, 7.0),
        Array(6.0, 2.0),
        Array(7.0, 5.0)
      ))
    }

    it("should have 10 features") {
      tiles.noFeatures shouldBe 10
    }

    it("should generate x features") {
      tiles.features(create(Array(0.0, 0.0))) should contain theSameElementsAs Seq(0L, 9, 8, 7, 6, 5, 4, 3)
      tiles.features(create(Array(0.125, 0.0))) should contain theSameElementsAs Seq(0L, 9, 8, 7, 6, 5, 4)
    }

    it("should generate y features") {
      tiles.features(create(Array(0.0, 0.125))) should contain theSameElementsAs Seq(0L, 9, 8, 7, 6, 4, 3)
    }


    it("should generate 1.1 features") {
      tiles.features(create(Array(1.0, 0.0))) should contain theSameElementsAs Seq(1L, 0, 9, 8, 7, 6, 5, 4)
      tiles.features(create(Array(0.0, 1.0))) should contain theSameElementsAs Seq(3L, 2, 1, 0, 9, 8, 7, 6)
      tiles.features(create(Array(1.0, 1.0))) should contain theSameElementsAs Seq(4L, 3, 2, 1, 0, 9, 8, 7)
      tiles.features(create(Array(1.9, 1.9))) should contain theSameElementsAs Seq(4L, 7, 6, 5, 3, 2, 1)
    }
  }

  describe("4D Tiles with 1 tile") {
    val tiles = Tiles(1, 1, 1)
    it("should have 16 tilings") {
      tiles.tilings shouldBe 16
    }
  }

  describe("3D Tiles with 1 tile") {
    val tiles = Tiles(1, 1, 1)
    it("should have 16 tilings") {
      tiles.tilings shouldBe 16
    }

    it("should have 3D offset") {
      tiles.offsets.mul(16) shouldBe create(Array(
        Array(0.0, 0.0, 0.0),
        Array(1.0, 3.0, 5.0),
        Array(2.0, 6.0, 10.0),
        Array(3.0, 9.0, 15.0),
        Array(4.0, 12.0, 4.0),
        Array(5.0, 15.0, 9.0),
        Array(6.0, 2.0, 14.0),
        Array(7.0, 5.0, 3.0),
        Array(8.0, 8.0, 8.0),
        Array(9.0, 11.0, 13.0),
        Array(10.0, 14.0, 2.0),
        Array(11.0, 1.0, 7.0),
        Array(12.0, 4.0, 12.0),
        Array(13.0, 7.0, 1.0),
        Array(14.0, 10.0, 6.0),
        Array(15.0, 13.0, 11.0)
      ))
    }

    it("should have 128 features") {
      tiles.noFeatures shouldBe 128
    }
  }
}

