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

package org.mmarini.scalarl.envs

import org.nd4j.linalg.factory.Nd4j
import org.scalatest.FunSpec
import org.scalatest.Matchers
import org.scalatest.prop.PropertyChecks

class MazeEnvTest extends FunSpec with PropertyChecks with Matchers {

  describe("Given an initial maze environment") {
    val data = List(
      "|  O       |",
      "|      X   |",
      "|  XXXXX   |",
      "|  X       |",
      "|    *     |")
    val env = MazeEnv.fromStrings(data)

    describe("When MazeEnv.reset") {
      val (_, obs) = env.reset()

      describe("Then observation") {
        val observation = obs.observation
        it("should be 1 at 1,0,4 ") {
          observation.getInt(1, 0, 4) should equal(1)
        }
        it("should be 1 at 0,1,2 ") {
          observation.getInt(0, 1, 2) should equal(1)
        }
        it("should be 1 at 0,3,6 ") {
          observation.getInt(0, 3, 6) should equal(1)
        }
      }
      describe("And actions") {
        val actions = obs.actions
        it("should be (1,1,1,0,0,0,1,1)") {
          actions should equal(Nd4j.create(Array(1, 1, 1, 0, 0, 0, 1, 1).map(_.toDouble)))
        }
      }
    }

    describe("When MazeEnv.step") {
      val (env1, obs, reward, endUp, info) = env.step(0)

      describe("Then observation") {
        val observation = obs.observation
        it("should be 0 at 1,0,4 ") {
          observation.getInt(1, 0, 4) should equal(0)
        }
        it("should be 1 at 1,1,4 ") {
          observation.getInt(1, 1, 4) should equal(1)
        }
        it("should be 1 at 0,1,2 ") {
          observation.getInt(0, 1, 2) should equal(1)
        }
        it("should be 1 at 0,3,6 ") {
          observation.getInt(0, 3, 6) should equal(1)
        }
      }
      describe("And actions") {
        val actions = obs.actions
        it("should be (0,0,1,1,1,1,1,0)") {
          actions should equal(Nd4j.create(Array(0, 0, 1, 1, 1, 1, 1, 0).map(_.toDouble)))
        }
      }
      describe("And reward") {
        it("should be -1") {
          reward should equal(-1.0)
        }
      }
      describe("And endUp") {
        it("should be false") {
          endUp should equal(false)
        }
      }
    }
  }

  describe("Given an initial maze environment with subject near wall") {
    val data = List(
      "|  O       |",
      "|      X   |",
      "|  XXXXX   |",
      "|  X*      |",
      "|          |")
    val env = MazeEnv.fromStrings(data)

    describe("When MazeEnv.reset") {
      val (_, obs) = env.reset()

      describe("Then observation") {
        val observation = obs.observation
        it("should be 1 at 1,1,3") {
          observation.getInt(1, 1, 3) should equal(1)
        }
        it("should be 1 at 0,1,2 ") {
          observation.getInt(0, 1, 2) should equal(1)
        }
        it("should be 1 at 0,3,6 ") {
          observation.getInt(0, 3, 6) should equal(1)
        }
      }
      describe("And actions") {
        val actions = obs.actions
        it("should be (0,0,1,1,1,1,0,0)") {
          actions should equal(Nd4j.create(Array(0, 0, 1, 1, 1, 1, 0, 0).map(_.toDouble)))
        }
      }
    }
  }

  describe("Given an initial maze environment with subject near target") {
    val data = List(
      "|          |",
      "|      X   |",
      "|  XXXXX   |",
      "|  X*O     |",
      "|          |")
    val env = MazeEnv.fromStrings(data)

    describe("When MazeEnv.reset") {
      val (_, obs) = env.reset()

      describe("Then observation") {
        val observation = obs.observation
        it("should be 1 at 1,1,3") {
          observation.getInt(1, 1, 3) should equal(1)
        }
        it("should be 1 at 0,1,2 ") {
          observation.getInt(0, 1, 2) should equal(1)
        }
        it("should be 1 at 0,3,6 ") {
          observation.getInt(0, 3, 6) should equal(1)
        }
      }
      describe("And actions") {
        val actions = obs.actions
        it("should be (0,0,1,1,1,1,0,0)") {
          actions should equal(Nd4j.create(Array(0, 0, 1, 1, 1, 1, 0, 0).map(_.toDouble)))
        }
      }
    }

    describe("When MazeEnv.step east") {
      val (env1, obs, reward, endUp, info) = env.step(2)

      describe("Then observation") {
        val observation = obs.observation
        it("should be 0 at 1,1,3") {
          observation.getInt(1, 1, 3) should equal(0)
        }
        it("should be 1 at 1,1,4") {
          observation.getInt(1, 1, 4) should equal(1)
        }
        it("should be 1 at 0,1,2 ") {
          observation.getInt(0, 1, 2) should equal(1)
        }
        it("should be 1 at 0,3,6 ") {
          observation.getInt(0, 3, 6) should equal(1)
        }
      }
      describe("And reward") {
        it("should be 1") {
          reward should equal(1.0)
        }
      }
      describe("And endUp") {
        it("should be true") {
          endUp should equal(true)
        }
      }
    }
  }
}
