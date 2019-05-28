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

import org.scalatest.Matchers
import org.scalatest.GivenWhenThen
import org.scalatest.prop.PropertyChecks
import org.scalatest.PropSpec
import org.scalatest.FunSpec

class MazeTest extends FunSpec with PropertyChecks with Matchers {

  describe("Given an empty string sequence") {
    val data = List()
    describe("When Maze.fromStrings") {
      it("should generate IllegalArgumentException") {
        the[IllegalArgumentException] thrownBy {
          Maze.fromStrings(data)
        } should have message "requirement failed: line 1: There must be at least one line"
      }
    }
  }

  describe("Given a string sequence of 2 chars") {
    val data = List("||")
    describe("When Maze.fromStrings") {
      it("should generate IllegalArgumentException") {
        the[IllegalArgumentException] thrownBy {
          Maze.fromStrings(data)
        } should have message "requirement failed: line 1: There must be at least 3 chars"
      }
    }
  }

  describe("Given a string sequence with wrong head") {
    val data = List("  |")
    describe("When Maze.fromStrings") {
      it("should generate IllegalArgumentException") {
        the[IllegalArgumentException] thrownBy {
          Maze.fromStrings(data)
        } should have message "requirement failed: line 1: Line must start with '|'"
      }
    }
  }

  describe("Given a string sequence with wrong tail") {
    val data = List(
      "|  |",
      "|   ")
    describe("When Maze.fromStrings") {
      it("should generate IllegalArgumentException") {
        the[IllegalArgumentException] thrownBy {
          Maze.fromStrings(data)
        } should have message "requirement failed: line 2: Line must end with '|'"
      }
    }
  }

  describe("Given a string sequence with wrong size") {
    val data = List(
      "|  |",
      "| |")
    describe("When Maze.fromStrings") {
      it("should generate IllegalArgumentException") {
        the[IllegalArgumentException] thrownBy {
          Maze.fromStrings(data)
        } should have message "requirement failed: line 2: Line must have 4 characters"
      }
    }
  }

  describe("Given a string sequence without initial chars") {
    val data = List(
      "|  O       |",
      "|          |",
      "|  XXXXX   |",
      "|          |",
      "|          |")
    describe("When Maze.fromStrings") {
      it("should generate IllegalArgumentException") {
        the[IllegalArgumentException] thrownBy {
          Maze.fromStrings(data)
        } should have message "requirement failed: line 6: File must contain a '*'"
      }
    }
  }

  describe("Given a string sequence with multiple initial chars") {
    val data = List(
      "|  O       |",
      "| *        |",
      "|  XXXXX   |",
      "|    *     |",
      "|          |")
    describe("When Maze.fromStrings") {
      it("should generate IllegalArgumentException") {
        the[IllegalArgumentException] thrownBy {
          Maze.fromStrings(data)
        } should have message "requirement failed: line 4: File must contain a single '*'"
      }
    }
  }

  describe("Given a string sequence without final chars") {
    val data = List(
      "|          |",
      "|          |",
      "|  XXXXX   |",
      "|          |",
      "|     *    |")
    describe("When Maze.fromStrings") {
      it("should generate IllegalArgumentException") {
        the[IllegalArgumentException] thrownBy {
          Maze.fromStrings(data)
        } should have message "requirement failed: line 6: File must contain a 'O'"
      }
    }
  }

  describe("Given a string sequence with multiple final chars") {
    val data = List(
      "|  O       |",
      "|          |",
      "|  XXXXX O |",
      "|          |",
      "|    *     |")
    describe("When Maze.fromStrings") {
      it("should generate IllegalArgumentException") {
        the[IllegalArgumentException] thrownBy {
          Maze.fromStrings(data)
        } should have message "requirement failed: line 3: File must contain a single 'O'"
      }
    }
  }

  describe("Given a valid sequence") {
    val data = List(
      "|  O       |",
      "|      X   |",
      "|  XXXXX   |",
      "|  X       |",
      "|    *     |")
    describe("When Maze.fromStrings") {
      val maze = Maze.fromStrings(data)

      it("should have 10 width") {
        maze.width should equal(10)
      }

      it("should have 5 height") {
        maze.height should equal(5)
      }

      it("should have intial at (4,4)") {
        maze.initial should equal(MazePos(4, 4))
      }

      it("should have target at (0,2)") {
        maze.target should equal(MazePos(0, 2))
      }

      it("should have expected walls") {
        maze.walls should equal(Array(
          false, false, false, false, false, false, false, false, false, false,
          false, false, false, false, false, false, true, false, false, false,
          false, false, true, true, true, true, true, false, false, false,
          false, false, true, false, false, false, false, false, false, false,
          false, false, false, false, false, false, false, false, false, false))
      }

    }
  }
}
