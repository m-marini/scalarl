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

import java.io.File

import com.typesafe.scalalogging.LazyLogging
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.util.ModelSerializer
import org.mmarini.scalarl.v2.Utils._
import org.mmarini.scalarl.v2._
import org.mmarini.scalarl.v2.agents.ACAgent._
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.api.rng.Random
import org.nd4j.linalg.ops.transforms.Transforms._

/**
 * Actor critic agent
 *
 * @param actor       the actor network
 * @param critic      the critic network
 * @param avg         the average rewards
 * @param alpha       the alpha parameter
 * @param rewardDecay the beta parameter
 * @param valueDecay  the value decay
 */
case class ACAgent(actor: MultiLayerNetwork,
                   critic: MultiLayerNetwork,
                   avg: INDArray,
                   alpha: INDArray,
                   rewardDecay: INDArray,
                   valueDecay: INDArray) extends Agent with LazyLogging {
  /**
   * Returns the new agent and the chosen action.
   * Chooses the action to be executed to the environment.
   *
   * @param observation the observation of environment
   * @param random      the random generator
   */
  override def chooseAction(observation: Observation, random: Random): Action = {
    val prefs = actor.output(observation.signals)
    val prefs1 = clip(prefs, -PreferenceRange, PreferenceRange)
    val pi = softmax(prefs)
    randomInt(pi)(random)
  }


  /**
   * Returns the fit agent by optimizing its strategy policy and the score
   *
   * @param feedback the feedback from the last step
   * @param random   the random generator
   */
  override def fit(feedback: Feedback, random: Random): (Agent, INDArray) = {
    val Feedback(s0, action, reward, s1) = feedback
    val v0 = v(s0)
    val v1 = v(s1)
    val (delta, newv0, newAvg) = computeDelta(v0, v1, reward, avg, valueDecay, rewardDecay)

    // Critic update
    val criticLabel = newv0
    val newCritic = critic.clone()
    newCritic.fit(s0.signals, criticLabel)
    val score = pow(delta, 2)

    // Actor update
    val prefs = actor.output(s0.signals)
    val actorLabel1 = computeActorLabel(prefs, action, alpha, delta)
    val newActor = actor.clone()
    newActor.fit(s0.signals, actorLabel1)
    logger.whenDebugEnabled {
      val pi = softmax(prefs)
      val npr = prefs.sub(prefs.mean())
      val v2 = v(newCritic, s0)
      val pi1 = softmax(actorLabel1)
      val pr2 = newActor.output(s0.signals)
      val pi2 = softmax(pr2)
      logger.debug("---------------------------------------------------------------")
      logger.debug("  s0      = {}", s0.signals)
      logger.debug("  action  = {}", action)
      logger.debug("  reward  = {}", reward)
      logger.debug("  s1      = {}", s1.signals)
      logger.debug("  delta   = {}", delta)
      logger.debug("  v0      = {}", v0)
      logger.debug("  v0'     = {}", newv0)
      logger.debug("  v0\"     = {}", v2)
      logger.debug("  v1      = {}", v1)
      logger.debug("  avg(R)  = {}", avg)
      logger.debug("  avg'(R) = {}", newAvg)
      logger.debug("  pr      = {} ", prefs)
      logger.debug("  pr'    = {}", actorLabel1)
      logger.debug("  pr\"     = {}", pr2)
      logger.debug("  pi      = {}", pi)
      logger.debug("  pi'     = {}", pi1)
      logger.debug("  pi\"     = {}", pi2)
      logger.debug("  score    = {}", newCritic.score())
    }
    val newAgent = copy(actor = newActor, critic = newCritic, avg = newAvg)
    (newAgent, score)
  }

  /**
   * Returns the value of a state
   *
   */
  private def v(obs: Observation): INDArray = v(critic, obs)

  /**
   * Returns the value of a state
   *
   * @param critic the critic
   * @param obs    the observation
   */
  private def v(critic: MultiLayerNetwork, obs: Observation): INDArray =
    critic.output(obs.signals)

  /**
   * Returns the score for a feedback
   *
   * @param feedback the feedback from the last step
   */
  override def score(feedback: Feedback): INDArray = {
    val Feedback(s0, _, reward, s1) = feedback
    val v0 = v(s0)
    val v1 = v(s1)
    val (delta, _, _) = computeDelta(v0, v1, reward, avg, valueDecay, rewardDecay)
    val score = pow(delta, 2)
    score
  }

  /**
   * Writes the agent status to file
   *
   * @param file the model folder
   * @return the agents
   */
  override def writeModel(file: String): Agent = {
    ModelSerializer.writeModel(actor, new File(file, "actor.zip"), false)
    ModelSerializer.writeModel(critic, new File(file, "critic.zip"), false)
    this
  }
}

object ACAgent {
  val PreferenceRange = 7

  /**
   * Returns delta, v0', avg'
   *
   * @param v0          the initial state value
   * @param v1          the final state value
   * @param reward      the reward
   * @param avg         the average reward
   * @param valueDecay  the value decay
   * @param rewardDecay the reward decay
   */
  def computeDelta(v0: INDArray,
                   v1: INDArray,
                   reward: INDArray,
                   avg: INDArray,
                   valueDecay: INDArray,
                   rewardDecay: INDArray): (INDArray, INDArray, INDArray) = {
    //    val newv0 = (v1 + reward - avg) * valueDecay + (1 - valueDecay) * avg
    val target = v1.add(reward).subi(avg)
    val delta = target.sub(v0)
    val newv0 = target.mul(valueDecay).subi(valueDecay.sub(1).muli(avg))
    //val newAvg = rewardDecay * avg + (1 - rewardDecay) * reward
    val newAvg = rewardDecay.mul(avg).subi(rewardDecay.sub(1).muli(reward))
    (delta, newv0, newAvg)
  }

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
    val actorLabel = prefs.add(deltaPref)

    // normalize
    val actorLabel1 = clip(actorLabel.sub(actorLabel.mean()), -PreferenceRange, PreferenceRange)
    actorLabel1
  }
}