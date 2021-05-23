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

package org.mmarini.scalarl.v6.agents

import io.circe.ACursor
import monix.reactive.subjects.PublishSubject
import org.mmarini.scalarl.v6.Configuration._
import org.mmarini.scalarl.v6.Utils._
import org.nd4j.linalg.api.ndarray.INDArray

import scala.util.{Failure, Try}

/**
 *
 * @param netInputDimensions     the number of network inputs
 * @param stateEncode            the inputs encoder
 * @param denormalizeActionValue the output denormalize function
 * @param normalizeActionValue   the output normalize function
 * @param valueDecay             the value decay parameter
 * @param rewardDecay            the reward decay parameter
 * @param actors                 the actor configuration
 * @param agentObserver          the agent event observer
 */
case class ActorCriticAgentConf(netInputDimensions: Int,
                                stateEncode: INDArray => INDArray,
                                denormalizeActionValue: INDArray => INDArray,
                                normalizeActionValue: INDArray => INDArray,
                                valueDecay: INDArray,
                                rewardDecay: INDArray,
                                actors: Seq[Actor],
                                agentObserver: PublishSubject[AgentEvent]) {

  /** Returns the number of network outputs */
  def noOutputs: Seq[Int] = 1 +: actors.map(_.noOutputs)
}

/**
 *
 */
object ActorCriticAgentConf {

  /**
   * Returns the agent configuration
   *
   * @param conf             the json configuration
   * @param stateDimensions  number of signal dimensions
   * @param actionDimensions number of action dimensions
   */
  def fromJson(conf: ACursor)(stateDimensions: Int, actionDimensions: Int):
  Try[(ActorCriticAgentConf, Seq[INDArray])] = {
    for {
      rewardDecay <- scalarFromJson(conf.downField("rewardDecay"))
      valueDecay <- scalarFromJson(conf.downField("valueDecay"))
      rewardRange <- rangesFromJson(conf.downField("rewardRange"))(1)
      (actors, alpha) <- actorsFromJson(conf.downField("actors"))(actionDimensions)
      (encode, noInputs) <- Encoder.fromJson(conf.downField("stateEncoder"))(stateDimensions)
    } yield {
      val conf = apply(netInputDimensions = noInputs,
        stateEncode = encode,
        valueDecay = valueDecay,
        rewardDecay = rewardDecay,
        rewardRange = rewardRange,
        actors = actors)
      (conf, alpha)
    }
  }

  /**
   * Returns an ActorCriticAgentConf
   *
   * @param rewardDecay        the reward decay parameter
   * @param valueDecay         the value decay parameter
   * @param rewardRange        the range of reward
   * @param actors             the actors
   * @param stateEncode        the state encode function
   * @param netInputDimensions the number of network input
   */
  def apply(rewardDecay: INDArray,
            valueDecay: INDArray,
            rewardRange: INDArray,
            actors: Seq[Actor],
            stateEncode: INDArray => INDArray,
            netInputDimensions: Int
           ): ActorCriticAgentConf = ActorCriticAgentConf(netInputDimensions = netInputDimensions,
    stateEncode = stateEncode,
    denormalizeActionValue = clipAndDenormalize(rewardRange),
    normalizeActionValue = clipAndNormalize(rewardRange),
    valueDecay = valueDecay,
    rewardDecay = rewardDecay,
    actors = actors,
    agentObserver = PublishSubject[AgentEvent]()
  )

  /**
   * Returns the list of actors and the alpha parameters
   *
   * @param conf             the json configuration
   * @param actionDimensions the number of actors
   */
  def actorsFromJson(conf: ACursor)(actionDimensions: Int): Try[(Seq[Actor], Seq[INDArray])] =
    Try {
      val seq = for {
        i <- 0 until actionDimensions
      } yield {
        actorFromJson(conf.downN(i))(i).get
      }
      seq.unzip
    }

  /**
   * Returns the agent and the alpha parameters
   *
   * @param conf      the json configuration
   * @param dimension the dimension index
   */
  def actorFromJson(conf: ACursor)(dimension: Int): Try[(Actor, INDArray)] = for {
    typ <- conf.get[String]("type").toTry
    actor <- typ match {
      case "PolicyActor" => PolicyActor.fromJson(conf)(dimension)
      case "GaussianActor" => GaussianActor.fromJson(conf)(dimension)
      case _ => Failure(new IllegalArgumentException(s"Actor $dimension '$typ' unrecognized"))
    }
  } yield actor
}
