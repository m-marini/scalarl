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

import scala.collection.Seq
import scala.util.Random

object RndMazeMain {

  val Lines = Seq(
    "|   O      |",
    "|          |",
    "|          |",
    "|          |",
    "|   XXX    |",
    "|          |",
    "|          |",
    "|          |",
    "|     *    |",
    "|          |")

  def main(args: Array[String]) {
    var (env, obs) = MazeEnv.fromStrings(Lines).reset();
    env.render()
    for { a <- 1 to 1000 } {
      val actions = obs.actions
      val actionIndices = for {
        i <- 0 until actions.size(0).toInt
        if (actions.getInt(i) > 0)
      } yield {
        i
      }
      val action = actionIndices(Random.nextInt(actionIndices.length))
      env.step(action) match {
        case (e, o, _, _, _) =>
          env = e
          obs = o
      }
      env.render()
      Thread.sleep(10)
    }
  }

}
