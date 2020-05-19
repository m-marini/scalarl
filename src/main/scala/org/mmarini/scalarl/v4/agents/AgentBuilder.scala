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

package org.mmarini.scalarl.v4.agents

import java.io.File

import com.typesafe.scalalogging.LazyLogging
import io.circe.ACursor
import monix.reactive.Observable
import monix.reactive.subjects.PublishSubject
import org.deeplearning4j.nn.graph.ComputationGraph
import org.deeplearning4j.util.ModelSerializer
import org.mmarini.scalarl.v4.{ActionConfig, Agent, DiscreteAction}
import org.nd4j.linalg.factory.Nd4j._

/**
 *
 */
object AgentBuilder extends LazyLogging {


  /**
   * Returns [[Agent]] from json configuration
   *
   * @param conf         the configuration
   * @param noInputs     the number of inputs
   * @param actionConfig actions configuration
   */
  def fromJson(conf: ACursor)(noInputs: Int, actionConfig: Seq[ActionConfig]): (Agent, Observable[AgentEvent]) = {
    val result = conf.get[String]("type").toTry.get match {
      case "ActorCritic" => actorCriticFromJson(conf)(noInputs, actionConfig)
      case typ =>
        throw new IllegalArgumentException(s"Agent type '$typ' unrecognized")
    }
    result
  }

  /**
   * Returns the [[ActorCriticAgent]] from json
   *
   * @param conf         the configuration
   * @param noInputs     the number of inputs
   * @param actionConfig the action configuration
   */
  def actorCriticFromJson(conf: ACursor)(noInputs: Int, actionConfig: Seq[ActionConfig]): (ActorCriticAgent, Observable[AgentEvent]) = {
    val modelPath = conf.get[String]("modelPath").toOption
    val avg = conf.get[Double]("avgReward").toTry.map(ones(1).muli(_)).get
    val rewardDecay = conf.get[Double]("rewardDecay").toTry.map(ones(1).muli(_)).get
    val valueDecay = conf.get[Double]("valueDecay").toTry.map(ones(1).muli(_)).get

    val actions = conf.downField("actors")
    val actors = actionConfig.zipWithIndex.map {
      case (action, dim) =>
        actorFromJson(actions.downN(dim))(dim, noInputs, action, modelPath)
    }

    val noOutputs = 1 +: actors.map(_.noOutputs)

    val network = modelPath.map(path => {
      loadNetwork(new File(path, s"network.zip"), noInputs, noOutputs)
    }).getOrElse(AgentNetworkBuilder.fromJson(conf.downField("network"))(noInputs, noOutputs))

    val plannerCfg = conf.downField("planner")
    val planner = if (plannerCfg.succeeded) Some(PriorityPlanner.fromJson(plannerCfg)(noInputs, actionConfig.length)) else None

    val subj = PublishSubject[AgentEvent]()
    val agent = ActorCriticAgent(actors = actors,
      network = network,
      avg = avg,
      valueDecay = valueDecay,
      rewardDecay = rewardDecay,
      planner = planner,
      agentObserver = subj
    )
    (agent, subj)
  }

  /**
   * Returns the agent
   *
   * @param dimension    the dimension index
   * @param conf         the json configuration
   * @param noInputs     the number ofr inputs
   * @param actionConfig the action configuration
   * @param modelPath    the path of model to load
   */
  def actorFromJson(conf: ACursor)(dimension: Int,
                                   noInputs: Int,
                                   actionConfig: ActionConfig,
                                   modelPath: Option[String]): Actor = {
    val typ = conf.get[String]("type").toTry.get
    (typ, actionConfig) match {
      case ("PolicyActor", cfg: DiscreteAction) =>
        policyActorFromJson(conf)(dimension, noInputs, cfg, modelPath)
      case ("GaussianActor", _) =>
        gaussianFromJson(conf)(dimension, noInputs, modelPath)
      case _ =>
        throw new IllegalArgumentException(s"Actor $dimension '$typ' incompatible with action configuration $actionConfig")
    }
  }

  /**
   * Returns the discrete action agent
   *
   * @param conf         the configuration element
   * @param dimension    the dimension index
   * @param noInputs     the number of inputs
   * @param actionConfig the actions configuration
   * @param modelPath    the path of model to load
   */
  def policyActorFromJson(conf: ACursor)(dimension: Int,
                                         noInputs: Int,
                                         actionConfig: DiscreteAction,
                                         modelPath: Option[String]): PolicyActor =
    PolicyActor(
      dimension = dimension,
      noOutputs = actionConfig.numValues,
      alpha = conf.get[Double]("alpha").toTry.map(ones(1).muli(_)).get)

  /**
   * Returns the discrete action agent
   *
   * @param conf      the configuration element
   * @param dimension the dimension index
   * @param noInputs  the number of inputs
   * @param modelPath the path of model to load
   */
  def gaussianFromJson(conf: ACursor)(dimension: Int,
                                      noInputs: Int,
                                      modelPath: Option[String]): GaussianActor = {
    val alphaMu = conf.get[Double]("alphaMu").toTry.get
    val alphaSigma = conf.get[Double]("alphaSigma").toTry.get
    GaussianActor(
      dimension = dimension,
      eta = create(Array(alphaMu, alphaSigma)))
  }

  /**
   * Returns the network loaded from a file
   *
   * @param file      the file to load
   * @param noInputs  the number of inputs
   * @param noOutputs the number of outputs
   */
  def loadNetwork(file: File, noInputs: Int, noOutputs: Seq[Int]): ComputationGraph = {
    logger.info("Loading {} ...", file)
    val net = ModelSerializer.restoreComputationGraph(file, true)
    // Validate
    net.layerInputSize(0) match {
      case x if x != noInputs =>
        throw new IllegalArgumentException(s"Network $file with wrong ($x) input number: expected $noInputs")
    }
    net.getNumOutputArrays match {
      case n if n != noOutputs.length =>
        throw new IllegalArgumentException(s"Network $file with wrong ($n) output layers: expected ${noOutputs.length}")
    }
    net
  }
}
