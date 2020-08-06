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
import org.mmarini.scalarl.v4.Agent
import org.mmarini.scalarl.v4.Utils._
import org.nd4j.linalg.factory.Nd4j._

import scala.util.{Failure, Success, Try}

/**
 *
 */
object AgentBuilder extends LazyLogging {


  /**
   * Returns the agent
   *
   * @param dimension the dimension index
   * @param conf      the json configuration
   * @param noInputs  the number ofr inputs
   * @param modelPath the path of model to load
   */
  def actorFromJson(conf: ACursor)(dimension: Int,
                                   noInputs: Int,
                                   modelPath: Option[String]): Try[Actor] = for {
    typ <- conf.get[String]("type").toTry
    actor <- typ match {
      case "PolicyActor" => PolicyActor.fromJson(conf)(dimension, noInputs, modelPath)
      case "GaussianActor" => GaussianActor.fromJson(conf)(dimension, noInputs, modelPath)
      case _ => Failure(new IllegalArgumentException(s"Actor $dimension '$typ' unrecognized"))
    }
  } yield actor


  /**
   * Returns [[Agent]] from json configuration
   *
   * @param conf             the configuration
   * @param noInputs         the number of inputs
   * @param actionDimensions actions space configuration
   */
  def fromJson(conf: ACursor)(noInputs: Int, actionDimensions: Int): Try[(Agent, Observable[AgentEvent])] = {
    val result = conf.get[String]("type").toTry.flatMap {
      case "ActorCritic" => actorCriticFromJson(conf)(noInputs, actionDimensions)
      case typ => Failure(new IllegalArgumentException(s"Agent type '$typ' unrecognized"))
    }
    result
  }

  def actorsFromJson(conf: ACursor)(noInputs: Int, actionDimensions: Int, modelPath: Option[String]): Try[Seq[Actor]] =
    Try {
      for {
        i <- 0 until actionDimensions
      } yield
        actorFromJson(conf.downN(i))(i, noInputs, modelPath).get
    }

  /**
   * Returns the [[ActorCriticAgent]] from json
   *
   * @param conf             the configuration
   * @param noInputs         the number of inputs
   * @param actionDimensions the action space dimensions
   */
  def actorCriticFromJson(conf: ACursor)(noInputs: Int, actionDimensions: Int): Try[(ActorCriticAgent, Observable[AgentEvent])] = {
    for {
      avg <- conf.get[Double]("avgReward").toTry.map(ones(1).muli(_))
      rewardDecay <- conf.get[Double]("rewardDecay").toTry.map(ones(1).muli(_))
      valueDecay <- conf.get[Double]("valueDecay").toTry.map(ones(1).muli(_))
      rewardRange <- conf.get[Array[Double]]("rewardRange").toTry.map(create).map(_.transpose())
      modelPath = conf.get[String]("modelPath").toOption
      actors <- actorsFromJson(conf.downField("actors"))(noInputs, actionDimensions, modelPath)
      noOutputs = 1 +: actors.map(_.noOutputs)
      network <- modelPath.map(path => {
        loadNetwork(new File(path, s"network.zip"), noInputs, noOutputs)
      }).getOrElse(AgentNetworkBuilder.fromJson(conf.downField("network"))(noInputs, noOutputs))
      plannerCfg = conf.downField("planner")
      planner <- if (plannerCfg.succeeded) {
        PriorityPlanner.fromJson(plannerCfg)(noInputs, noOutputs.sum).map(Some(_))
      } else {
        Success(None)
      }
    } yield {
      val subj = PublishSubject[AgentEvent]()
      val agent = ActorCriticAgent(actors = actors,
        network = network,
        avg = avg,
        valueDecay = valueDecay,
        rewardDecay = rewardDecay,
        denormalize = denormalize(rewardRange),
        normalizer = normalize(rewardRange),
        planner = planner,
        agentObserver = subj
      )
      (agent, subj)
    }
  }

  /**
   * Returns the network loaded from a file
   *
   * @param file      the file to load
   * @param noInputs  the number of inputs
   * @param noOutputs the number of outputs
   */
  def loadNetwork(file: File, noInputs: Int, noOutputs: Seq[Int]): Try[ComputationGraph] = {
    logger.info("Loading {} ...", file)
    val net = ModelSerializer.restoreComputationGraph(file, true)
    // Validate
    Try {
      net.layerInputSize(0) match {
        case x if x != noInputs =>
          throw new IllegalArgumentException(s"Network $file with wrong ($x) input number: expected $noInputs")
      }
      net.getNumOutputArrays match {
        case n if n != noOutputs.length =>
          throw new IllegalArgumentException(s"Network $file with wrong ($n) output layers: expected ${
            noOutputs.length
          }")
      }
      net
    }
  }
}
