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

package org.mmarini.scalarl.ts.envs

import com.typesafe.scalalogging.LazyLogging
import org.mmarini.scalarl.agents.AgentNetworkBuilder
import org.mmarini.scalarl.ts.agents.DynaQPlusAgent

/**
 *
 */
object Main extends LazyLogging {

  /**
   *
   * @param args the line command arguments
   */
  def main(args: Array[String]) {
    val file = if (args.isEmpty) "maze.yaml" else args(0)
    logger.info("File {}", file)

    val jsonConf = Configuration.jsonFromFile(file)
    val env = EnvBuilder(jsonConf.hcursor.downField("env")).build()
    val net = AgentNetworkBuilder(jsonConf.hcursor.downField("network"),
      env.signalSize,
      env.actionConfig.sum).build()
    val agent = DynaQPlusAgent(jsonConf.hcursor.downField("agent"), net, env.actionConfig)
    val (session, random) = SessionBuilder(jsonConf.hcursor.downField("session")).
      build(env = env, agent = agent)

    session.run(random)
    logger.info("Session completed.")
  }
}