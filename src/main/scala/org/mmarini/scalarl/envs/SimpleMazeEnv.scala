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
import org.mmarini.scalarl.EndUp
import org.mmarini.scalarl.Env
import org.mmarini.scalarl.INDArrayObservation
import org.mmarini.scalarl.Observation
import org.mmarini.scalarl.Reward
import org.nd4j.linalg.factory.Nd4j;

/**
 * The environment simulating a subject in a maze.
 *
 * - Reseting the environment the subject is moved in the initial cell.
 * - The subject is moved according the 8 adjacent cell chose to move and returning a -1 point of reward.
 * - when the subject reach the target cell the episode end with a one point of reward.
 * - If the destination cell is outside the maze or is a wall the subject is not moved.
 *
 * @constructor Creates a [[MazeEnv]]
 * @param maze the maze
 * @param subject the current subject position
 */
case class SimpleMazeEnv(
  maze:    Maze,
  subject: MazePos) extends Env {
  private val TargetReward = 10.0
  private val NoStepReward = -2
  private val UnitMoveReward = -1.0

  private val Deltas: Array[(Int, Int)] = Array(
    (-1, 0), //N
    (-1, 1), // NE
    (0, 1), // E
    (1, 1), // SE
    (1, 0), // S
    (1, -1), // SW
    (0, -1), // W
    (-1, -1) // NW
  )

  def render() {
    val map = for {
      row <- 0 until maze.height
    } yield {
      val line = for {
        col <- 0 until maze.width
        pos = MazePos(row, col)
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
    print(map.mkString("\n"))
  }

  override def reset(): (Env, Observation) = {
    val next = SimpleMazeEnv(
      maze = maze,
      subject = maze.initial)
    val obs = next.observation
    (next, obs)
  }

  private def endUp(): Boolean = maze.isTarget(subject)

  private def moveCost(pos: MazePos): Double =
    if (subject == pos) NoStepReward else subject.distance(pos) * UnitMoveReward

  override def step(action: Action): (Env, Observation, Reward, EndUp) = {
    val delta = if (action >= 0 && action < Deltas.length) Deltas(action) else (0, 0)
    val destination = subject.moveBy(delta)
    val pos = if (maze.isValid(destination)) destination else subject
    val endUp = maze.isTarget(pos)
    val reward = moveCost(pos) + (if (endUp) TargetReward else 0.0)
    val nextEnv = copy(subject = pos)
    (nextEnv, nextEnv.observation, reward, endUp)
  }

  /** Returns the observation for a given subject location */
  private def observation(subject: MazePos): Observation = {
    // Computes the available actions
    val actions = Nd4j.zeros(Deltas.length.toLong)
    for {
      (delta, action) <- Deltas.zipWithIndex
      pos = subject.moveBy(delta)
      if maze.isValid(pos)
    } {
      actions.putScalar(action, 1)
    }

    // Computes the environment status
    // array of 2 x widht x height
    val shape = maze.map.shape()
    val observation = Nd4j.zeros(shape: _*)

    // Fills with subject position
    observation.putScalar(Array(subject.row, subject.col), 1)

    val obs = INDArrayObservation(observation = observation.ravel(), actions = actions.ravel())
    obs
  }

  lazy val observation: Observation = observation(subject)

  /**
   * Returns the sequence of observation for each of possible states
   * Each observation consists of an array of states and an array of available actions.
   * The state array consists of 10 x 10 values 1 value for the cell with subject
   * and a second sequence of 10 x 10 values with 1 value foe the cell with obstacle
   * The action array contains 9 values each for any direction with 1 value if action is valid
   */
  def dumpStates: Seq[Observation] = for {
    row <- 0 until maze.height
    col <- 0 until maze.width
  } yield {
    observation(MazePos(row, col))
  }

}

object SimpleMazeEnv {
  def fromStrings(lines: Seq[String]): SimpleMazeEnv = {
    val maze = Maze.fromStrings(lines)
    SimpleMazeEnv(maze, maze.initial)
  }
}
