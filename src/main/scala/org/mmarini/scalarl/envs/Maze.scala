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

import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j

/**
 * The Maze with walls, initial subject position and target position
 *
 * @constructor Create a Maze with width and height, wall map, initial subject position and taregt position
 * @param width the width of maze
 * @param height the height of maze
 * @param walls the wall map
 * @param initial the initial position of subject
 * @param target the target position
 */
case class Maze(
  width:   Int,
  height:  Int,
  walls:   Array[Boolean],
  initial: MazePos,
  target:  MazePos) {

  /** Returns the flat position index */
  def index(pos: MazePos): Int = pos.x + pos.y * width

  /** Returns true if the position is a wall */
  def isWall(pos: MazePos): Boolean = walls(index(pos))

  /** Returns true if the position is a target */
  def isTarget(pos: MazePos): Boolean = pos == target

  /** Returns true if the position is outer the maze*/
  def isOuter(pos: MazePos): Boolean =
    pos.x < 0 || pos.x >= width || pos.y < 0 || pos.y >= height

  /** Returns true if the position is valid */
  def isValid(pos: MazePos): Boolean = !(isOuter(pos) || isWall(pos))

  /** Returns the map of wall in the maze */
  lazy val map: INDArray = {
    val observation = Nd4j.zeros(Array(height, width), 'c')
    for {
      x <- 0 until width
      y <- 0 until height
      pos = MazePos(x, y)
      if !isValid(pos)
    } {
      observation.putScalar(Array(y, x), 1)
    }
    observation
  }
}

/** Factory for [[Maze]] instances */
object Maze {

  /** Creates a [[Maze]] by parsing a list of lines representing the environment */
  def fromStrings(lines: Seq[String]): Maze = {
    val (w, h) = size(lines)
    val i = MazePos(position(lines, '*'))
    val t = MazePos(position(lines, 'O'))
    val ws = walls(lines)
    Maze(
      width = w,
      height = h,
      walls = ws,
      initial = i,
      target = t)
  }

  /** Returns the width and height of maze by parsing the list of lines */
  private def size(lines: Seq[String]): (Int, Int) = {
    val height = lines.length
    require(height > 0, "line 1: There must be at least one line")
    val width = lines.head.length() - 2
    require(width > 0, "line 1: There must be at least 3 chars")
    for {
      (row, i) <- lines.zipWithIndex
    } {
      require(row.head == '|', s"line ${i + 1}: Line must start with '|'")
      require(row.last == '|', s"line ${i + 1}: Line must end with '|'")
      require(row.length() == width + 2, s"line ${i + 1}: Line must have ${width + 2} characters")
    }
    (width, height)
  }

  /** Returns the position definition by parsing the list of lines */
  private def position(lines: Seq[String], target: Char): (Int, Int) = {
    val indices = for {
      (line, row) <- lines.zipWithIndex
      col = line.indexOf(target)
      if (col > 0)
    } yield (col - 1, row)
    require(!indices.isEmpty, s"line ${lines.length + 1}: File must contain a '${target}'")
    require(indices.length == 1, s"line ${indices(1)._2 + 1}: File must contain a single '${target}'")
    indices.head match {
      case (x, y) => (x, lines.length - y - 1)
    }
  }

  /** Returns the wall map by parsing the list of lines */
  private def walls(lines: Seq[String]): Array[Boolean] = {
    val height = lines.length
    val width = lines.head.length() - 2
    val walls = for {
      row <- lines
    } yield {
      row.tail.init.map(_ == 'X')
    }
    walls.reverse.flatten.toArray
  }
}
