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

package org.mmarini.scalarl.v2.agents

import io.circe.ACursor
import org.mmarini.scalarl.v2.Agent

/**
 *
 */
object AgentBuilder {
  /**
   * Returns the agent
   *
   * @param conf      the json configuration
   * @param noInputs  the number ofr inputs
   * @param noOutputs the number of outpus
   */
  def fromJson(conf: ACursor)(noInputs: Int, noOutputs: Int): Agent = conf.get[String]("type").right.get match {
    case "ExpectedSarsaAgent" => ExpSarsaAgent.fromJson(conf)(noInputs, noOutputs)
    case "ActorCriticAgent" => acAgent(conf)(noInputs, noOutputs)
    case _ => throw new IllegalArgumentException("Wrong agent type")
  }

  /**
   * Returns the Dyna+Agent
   *
   * @param conf the configuration element
   */
  def acAgent(conf: ACursor)(noInputs: Int, noOutputs: Int): ACAgent = {
    val netConf = conf.downField("network")
    ACAgent(
      actor = AgentNetworkBuilder.fromJson(netConf)(noInputs, noOutputs),
      critic = AgentNetworkBuilder.fromJson(netConf)(noInputs, 1),
      alpha = conf.get[Double]("alpha").right.get,
      beta = conf.get[Double]("beta").right.get,
      actorRatio = conf.get[Double]("actorRatio").right.get,
      criticRatio = conf.get[Double]("criticRatio").right.get,
      avg = conf.get[Double]("avgReward").right.get)
  }
}
