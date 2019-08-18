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
import org.mmarini.scalarl.ChannelAction

class MazeEnvTest extends FunSpec with PropertyChecks with Matchers {

  def action(idx: Int): ChannelAction = {
    val action = Nd4j.zeros(8)
    action.putScalar(idx, 1)
    action
  }

  describe("Given an initial maze environment") {
    val data = List(
      "|  O       |",
      "|      X   |",
      "|  XXXXX   |",
      "|  X       |",
      "|     *    |")
    val env = MazeEnv.fromStrings(data)

    describe("When MazeEnv.reset") {
      val (env0, obs) = env.reset()

      describe("Then observation") {
        val observation = obs.signals
        it("should be 1 at subject position=0,4,5, index=40+5=45") {
          observation.getInt(45) should equal(1)
        }
      }
      describe("And valid actions") {
        val actions = obs.actions
        it("should be (1,1,1,0,0,0,1,1)") {
          val expected = Nd4j.create(Array(1, 1, 1, 0, 0, 0, 1, 1).map(_.toDouble)).ravel()
          actions should equal(expected)
        }
      }
    }

    describe("When MazeEnv.step north west") {
      val (env1, obs, reward) = env.step(action(7))

      describe("Then observation") {
        val observation = obs.signals
        it("should be 0 at subject position=0,4,5, index=40+5=45") {
          observation.getInt(44) should equal(0)
        }
        it("should be 1 at subject position=0,3,4, index=30+4=34") {
          observation.getInt(34) should equal(1)
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
          reward should equal(-3.414 +- 0.001)
        }
      }
      describe("And endUp") {
        it("should be false") {
          obs.endUp should equal(false)
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
        val observation = obs.signals
        it("should be 1 at subject position=0,3,3, index=30+3=3") {
          observation.getInt(33) should equal(1)
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
        val observation = obs.signals
        it("should be 1 at subject position=0,3,3, index=30+3=3") {
          observation.getInt(33) should equal(1)
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
      val (env1, obs, reward) = env.step(action(2))

      describe("Then observation") {
        val observation = obs.signals
        it("should be 0 at subject position=0,3,3, index=30+3=33") {
          observation.getInt(33) should equal(0)
        }
        it("should be 1 at subject position=0,3,4, index=30+4=34") {
          observation.getInt(34) should equal(1)
        }
      }
      describe("And reward") {
        it("should be 7") {
          reward should equal(7.0)
        }
      }
      describe("And endUp") {
        it("should be true") {
          obs.endUp should equal(true)
        }
      }
    }
  }
}
