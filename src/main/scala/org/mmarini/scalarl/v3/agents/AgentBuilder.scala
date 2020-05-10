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

package org.mmarini.scalarl.v3.agents

import java.io.File

import com.typesafe.scalalogging.LazyLogging
import io.circe.ACursor
import org.deeplearning4j.nn.graph.ComputationGraph
import org.deeplearning4j.util.ModelSerializer
import org.mmarini.scalarl.v3.{ActionConfig, Agent, DiscreteAction}
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.factory.Nd4j.ones

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
  def fromJson(conf: ACursor)(noInputs: Int, actionConfig: Seq[ActionConfig]): Agent = {
    val agent = conf.get[String]("type").toTry.get match {
      case "ActorCritic" => actorCriticFromJson(conf)(noInputs, actionConfig)
      case "ExpSarsa" => expSarsaAgentFromJson(conf)(noInputs, actionConfig)
      case typ =>
        throw new IllegalArgumentException(s"Agent type '$typ' unrecognized")
    }
    agent
  }

  /**
   * Returns the [[ActorCriticAgent]] from json
   *
   * @param conf         the configuration
   * @param noInputs     the number of inputs
   * @param actionConfig the action configuration
   */
  def actorCriticFromJson(conf: ACursor)(noInputs: Int, actionConfig: Seq[ActionConfig]): ActorCriticAgent = {
    val modelPath = conf.get[String]("modelPath").toOption
    val avg = conf.get[Double]("avgReward").toTry.map(Nd4j.ones(1).muli(_)).get
    val rewardDecay = conf.get[Double]("rewardDecay").toTry.map(Nd4j.ones(1).muli(_)).get
    val valueDecay = conf.get[Double]("valueDecay").toTry.map(Nd4j.ones(1).muli(_)).get

    val actions = conf.downField("actors")
    val actors = actionConfig.zipWithIndex.map {
      case (action, dim) =>
        actorFromJson(actions.downN(dim))(dim, noInputs, action, modelPath)
    }

    val critic = modelPath.map(path => {
      loadNetwork(new File(path, s"critic.zip"), noInputs, 1)
    }).getOrElse(AgentNetworkBuilder.fromJson(conf.downField("critic"))(noInputs, 1, Activation.IDENTITY))

    val plannerCfg = conf.downField("planner")
    val planner = if (plannerCfg.succeeded) Some(PriorityPlanner.fromJson(plannerCfg)) else None

    ActorCriticAgent(actors = actors.toArray,
      critic = critic,
      avg = avg,
      valueDecay = valueDecay,
      rewardDecay = rewardDecay,
      planner = planner)
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
      case ("PolicyActor", DiscreteAction(nValues)) =>
        policyActorFromJson(conf)(dimension, noInputs, nValues, modelPath)
      case ("GaussianActor", _) =>
        gaussianFromJson(conf)(dimension, noInputs, modelPath)
      case _ =>
        throw new IllegalArgumentException(s"Actor $dimension '$typ' incompatible with action configuration $actionConfig")
    }
  }

  /**
   * Returns the discrete action agent
   *
   * @param conf      the configuration element
   * @param dimension the dimension index
   * @param noInputs  the number of inputs
   * @param noOutputs the number of outputs
   * @param modelPath the path of model to load
   */
  def policyActorFromJson(conf: ACursor)(dimension: Int,
                                         noInputs: Int,
                                         noOutputs: Int,
                                         modelPath: Option[String]): PolicyActor = {
    val netConf = conf.downField("network")
    val actor = modelPath.map(path => {
      loadNetwork(new File(path, s"actor-$dimension.zip"), noInputs, noOutputs)
    }).getOrElse(AgentNetworkBuilder.fromJson(netConf)(noInputs, noOutputs, Activation.IDENTITY))
    PolicyActor(
      dimension = dimension,
      actor = actor,
      alpha = conf.get[Double]("alpha").toTry.map(Nd4j.ones(1).muli(_)).get)
  }

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
    val netConf = conf.downField("network")
    val actor = modelPath.map(path => {
      loadNetwork(new File(path, s"actor-$dimension.zip"), noInputs, 2)
    }).getOrElse(AgentNetworkBuilder.fromJson(netConf)(noInputs, 2, Activation.IDENTITY))

    GaussianActor(
      dimension = dimension,
      actor = actor,
      alpha = conf.get[Double]("alpha").toTry.map(Nd4j.ones(1).muli(_)).get)
  }

  /**
   * Returns the [[ExpSarsaAgent]]
   *
   * @param conf         the json configuration
   * @param noInputs     the number of inputs
   * @param actionConfig the action configuration
   */
  def expSarsaAgentFromJson(conf: ACursor)(noInputs: Int, actionConfig: Seq[ActionConfig]): ExpSarsaAgent = {
    val actions = conf.downField("actions")
    val modelPath = conf.get[String]("modelPath").toOption
    val actionAgents = actionConfig.zipWithIndex.map {
      case (DiscreteAction(noActions), dim) =>
        expSarsaFromJson(actions.downN(dim))(dim, noInputs, noActions, modelPath)
      case (actionConf, dim) =>
        throw new IllegalArgumentException(s"Action $dim '$actionConf' incompatible with ExpSarsaAgent")
    }
    ExpSarsaAgent(actionAgents.toArray)
  }

  /**
   * Returns the ExpSarsaAgent
   *
   * @param dimension the dimension index
   * @param conf      the configuration element
   * @param noInputs  the neural inputs
   * @param noActions the number of action
   * @param modelPath the path of model to load
   */
  def expSarsaFromJson(conf: ACursor)(dimension: Int,
                                      noInputs: Int,
                                      noActions: Int,
                                      modelPath: Option[String]): ExpSarsaMethod = {
    val netConf = conf.downField("network")
    val net = modelPath.map(path => {
      loadNetwork(new File(path, s"network-$dimension.zip"), noInputs, noActions)
    }).getOrElse(AgentNetworkBuilder.fromJson(netConf)(noInputs, noActions, Activation.IDENTITY))

    val avgReward = conf.get[Double]("avgReward").toTry.map(
      ones(1).muli(_)
    ).get
    ExpSarsaMethod(dimension = dimension,
      net = net,
      noActions = noActions,
      beta = conf.get[Double]("beta").toTry.map(ones(1).muli(_)).get,
      avgReward = avgReward,
      epsilon = conf.get[Double]("epsilon").toTry.map(ones(1).muli(_)).get)
  }

  /**
   * Returns the network loaded from a file
   *
   * @param file      the file to load
   * @param noInputs  the number of inputs
   * @param noOutputs the number of outputs
   */
  def loadNetwork(file: File, noInputs: Int, noOutputs: Int): ComputationGraph = {
    logger.info("Loading {} ...", file)
    val net = ModelSerializer.restoreComputationGraph(file, true)
    // Validate
    net.layerInputSize(0) match {
      case x if x != noInputs =>
        throw new IllegalArgumentException(s"Network $file with wrong ($x) input number: expected $noInputs")
    }
    //    net.numLabels() match {
    //      case x if x != noOutputs =>
    //        throw new IllegalArgumentException(s"Network $file with wrong ($x) outputs number: expected $noOutputs")
    //    }
    net
  }
}
