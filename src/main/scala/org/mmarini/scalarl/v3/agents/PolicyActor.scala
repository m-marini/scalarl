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
import org.deeplearning4j.nn.graph.ComputationGraph
import org.deeplearning4j.util.ModelSerializer
import org.mmarini.scalarl.v3.Utils._
import org.mmarini.scalarl.v3._
import org.mmarini.scalarl.v3.agents.PolicyActor._
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.api.rng.Random
import org.nd4j.linalg.factory.Nd4j._
import org.nd4j.linalg.ops.transforms.Transforms._

/**
 * Actor critic agent
 *
 * @param dimension the dimension index
 * @param actor     the actor network
 * @param alpha     the alpha parameter
 */
case class PolicyActor(dimension: Int,
                       actor: ComputationGraph,
                       alpha: INDArray) extends Actor with LazyLogging {
  /**
   * Returns the new agent and the chosen action.
   * Chooses the action to be executed to the environment.
   *
   * @param observation the observation of environment
   * @param random      the random generator
   */
  override def chooseAction(observation: Observation, random: Random): INDArray = {
    val pi = softmax(preferences(observation))
    val action = ones(1).muli(randomInt(pi)(random))
    action
  }

  /**
   * Returns the preferences
   *
   * @param observation the observation
   */
  def preferences(observation: Observation): INDArray =
    normalize(actor.output(observation.signals)(0))

  /**
   * Returns the changed actor
   * The lerning method is applied to create a new fit actor
   *
   * @param feedback the feedback
   * @param delta    the TD Error
   * @param random   the random generator
   */
  override def fit(feedback: Feedback, delta: INDArray, random: Random): Actor = {
    val Feedback(s0, actions, reward, s1) = feedback
    val action = actions.getInt(dimension)

    // Actor update
    val prefs = preferences(s0)
    val actorLabel1 = computeActorLabel(prefs, action, alpha, delta)
    val newNet = actor.clone()
    newNet.fit(Array(s0.signals), Array(actorLabel1))
    val newActor = copy(actor = newNet)
    newActor
  }

  /**
   * Writes the agent status to file
   *
   * @param path the model folder
   * @return the agents
   */
  override def writeModel(path: File): Actor = {
    ModelSerializer.writeModel(actor, new File(path, s"actor-$dimension.zip"), false)
    this
  }
}

object PolicyActor {
  val PreferenceRange = 7

  /**
   * Returns the actor target label
   *
   * @param prefs  the preferences
   * @param action the action
   * @param alpha  the alpha parameter
   * @param delta  the TD Error
   */
  def computeActorLabel(prefs: INDArray, action: Int, alpha: INDArray, delta: INDArray): INDArray = {
    val pi = softmax(prefs)
    val expTot = exp(prefs).sum()
    // deltaH = (A_i(a) / expTot - pi) alpha delta
    val A = features(Seq(action), prefs.length()).divi(expTot)
    val deltaPref = A.sub(pi).muli(alpha).muli(delta)
    val actorLabel = normalize(prefs.add(deltaPref))
    actorLabel
  }

  /**
   * Returns the normalized preferences
   *
   * @param data the preferences
   */
  def normalize(data: INDArray): INDArray = clip(data.sub(data.mean()), -PreferenceRange, PreferenceRange)
}