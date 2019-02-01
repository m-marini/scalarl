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

import org.mmarini.scalarl.Action
import org.mmarini.scalarl.Env
import org.mmarini.scalarl.Observation
import org.mmarini.scalarl.EndUp
import org.mmarini.scalarl.Info
import org.mmarini.scalarl.Reward
import org.mmarini.scalarl.INDArrayObservation
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.indexing.NDArrayIndex
import org.mmarini.scalarl.Feedback

/**
 *
 */
case class MazeEnv(maze: Maze, subject: MazePos) extends Env {
  private val OuterReward = -1.0
  private val WallReward = -1.0
  private val TargetReward = 1.0
  private val NoStepReward = -1.0
  private val StepReward = -1.0

  private val ClearScreen = "\033[2J"

  private val Deltas: Array[(Int, Int)] = Array(
    (0, 1), //N
    (1, 1), // NE
    (1, 0), // E
    (1, -1), // SE
    (0, -1), // S
    (-1, -1), // SW
    (-1, 0), // W
    (-1, 1) // NW
  )

  override def render(mode: String, close: Boolean): Env = if (mode == "human") {
    val map = for {
      y <- 0 until maze.height
    } yield {
      val line = for {
        x <- 0 until maze.width
        pos = MazePos(x, y)
      } yield {
        if (pos == subject) {
          '*'
        } else if (pos == maze.target) {
          'O'
        } else if (maze.isValid(pos)) {
          ' '
        } else {
          'X'
        }
      }
      "|" + line.mkString + "|"
    }

    val txt = ClearScreen + map.reverse.mkString("\n")
    println(txt)
    this
  } else {
    this
  }

  override def reset(): (Env, Observation) = {
    val next = MazeEnv(maze = maze, subject = maze.initial)
    val obs = observation
    (next, obs)
  }

  override def step(action: Action): (Env, Observation, Reward, EndUp, Info) = {
    val delta = if (action >= 0 && action < Deltas.length) Deltas(action) else (0, 0)
    val nextPos = subject.moveBy(delta)
    if (!maze.isValid(nextPos)) {
      // Invalid
      (this, observation, NoStepReward, false, Map())
    } else if (nextPos.equals((0, 0))) {
      // No move
      (this, observation, NoStepReward, false, Map())
    } else if (maze.isTarget(nextPos)) {
      val next = moveTo(nextPos)
      (next, next.observation, TargetReward, true, Map())
    } else {
      val next = moveTo(nextPos)
      (next, next.observation, StepReward, false, Map())
    }
  }

  private lazy val observation: Observation = {
    val actions = Nd4j.zeros(Array(Deltas.length), 'c')
    for {
      (delta, action) <- Deltas.zipWithIndex
      pos = subject.moveBy(delta)
      if maze.isValid(pos)
    } {
      actions.putScalar(action, 1)
    }

    val shape = 2L +: maze.map.shape()
    val observation = Nd4j.zeros(shape, 'c')

    val int0 = NDArrayIndex.interval(0, 1)
    val int1 = NDArrayIndex.all()
    observation.get(int0, int1, int1).assign(maze.map)
    observation.putScalar(Array(1, subject.y, subject.x), 1)

    INDArrayObservation(observation = observation, actions = actions)
  }

  private def moveTo(nextPos: MazePos): MazeEnv = copy(subject = nextPos)

}

object MazeEnv {
  def fromStrings(lines: Seq[String]): MazeEnv = {
    val maze = Maze.fromStrings(lines)
    MazeEnv(maze, maze.initial)
  }

}
