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
import org.mmarini.scalarl.v2._
import org.mmarini.scalarl.v2.agents.ACContinuousActionAgent._
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.api.rng.Random
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.ops.transforms.Transforms

/**
 * Actor critic agent
 *
 * @param actor       the actor network for mu
 * @param critic      the critic network
 * @param avg         the average rewards
 * @param alpha       the alpha parameter
 * @param rewardDecay the beta parameter
 * @param valueDecay  the value decay
 */
case class ACContinuousActionAgent(actor: MultiLayerNetwork,
                                   critic: MultiLayerNetwork,
                                   avg: INDArray,
                                   alpha: INDArray,
                                   rewardDecay: INDArray,
                                   valueDecay: INDArray) extends AgentContinuousAction with LazyLogging {

  /**
   * Returns the new agent and the chosen action.
   * Chooses the action to be executed to the environment.
   *
   * @param observation the observation of environment
   * @param random      the random generator
   */
  override def chooseAction(observation: Observation, random: Random): INDArray = {
    val (mu, _, sigma) = muHSigma(observation)
    val action = sigma.mul(random.nextGaussian()).addi(mu)
    action
  }

  /**
   * Returns the fit agent by optimizing its strategy policy and the score
   *
   * @param feedback the feedback from the last step
   * @param random   the random generator
   */
  override def fit(feedback: FeedbackContinuousAction, random: Random): (AgentContinuousAction, INDArray) = {
    val FeedbackContinuousAction(s0, action, reward, s1) = feedback
    val v0 = v(s0)
    val v1 = v(s1)
    val (delta, newv0, newAvg) = computeDelta(v0, v1, reward, avg, rewardDecay, valueDecay)

    // Critic update
    val newCritic = critic.clone()
    newCritic.fit(s0.signals, newv0)

    // Actor update
    val (mu, h, sigma) = muHSigma(s0)

    val (mu1, h1) = computeActorTarget(action, alpha, delta, mu, h, sigma)

    val actorLabels = Nd4j.hstack(mu1, h1)
    val newActor = actor.clone()
    newActor.fit(s0.signals, actorLabels)

    logger.whenDebugEnabled {
      val sigma1 = Transforms.exp(h1)
      val v2 = newCritic.output(s0.signals)
      val (mu2, h2, sigma2) = muHSigma(s0)
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
      logger.debug("  mu      = {} ", mu)
      logger.debug("  mu'     = {}", mu1)
      logger.debug("  mu\"     = {}", mu2)
      logger.debug("  sigma    = {}", sigma)
      logger.debug("  sigma'   = {}", sigma1)
      logger.debug("  sigma\"   = {}", sigma2)
      logger.debug("  h        = {}", h)
      logger.debug("  h'       = {}", h1)
      logger.debug("  h\"       = {}", h2)
    }


    val newAgent = copy(actor = newActor, critic = newCritic, avg = newAvg)
    val score = delta.mul(delta)
    (newAgent, score)
  }

  /**
   * Returns mu, h, sigma
   *
   * @param s0 the state observation
   */
  def muHSigma(s0: Observation): (INDArray, INDArray, INDArray) = ACContinuousActionAgent.muHSigma(actor.output(s0.signals))

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
  private def v(critic: MultiLayerNetwork, obs: Observation): INDArray = critic.output(obs.signals)

  /**
   * Returns the score for a feedback
   *
   * @param feedback the feedback from the last step
   */
  override def score(feedback: FeedbackContinuousAction): INDArray = {
    val FeedbackContinuousAction(s0, _, reward, s1) = feedback
    val v0 = v(s0)
    val v1 = v(s1)
    val (delta, _, _) = computeDelta(v0, v1, reward, avg, rewardDecay, valueDecay)
    val score = delta.muli(delta)
    score
  }

  /**
   * Writes the agent status to file
   *
   * @param file the model folder
   * @return the agents
   */
  override def writeModel(file: String): AgentContinuousAction = {
    ModelSerializer.writeModel(actor, new File(file, "actor.zip"), false)
    ModelSerializer.writeModel(critic, new File(file, "critic.zip"), false)
    this
  }
}

/**
 *
 */
object ACContinuousActionAgent {
  val MuRange = 1e100
  val HRange = 100.0

  /**
   *
   * @param actor       the actor network
   * @param critic      the critic network
   * @param avg         the avg reward
   * @param alpha       the alpha parameter
   * @param rewardDecay the reward decay
   * @param valueDecay  the value decay
   */
  def apply(actor: MultiLayerNetwork,
            critic: MultiLayerNetwork,
            avg: Double,
            alpha: Double,
            rewardDecay: Double,
            valueDecay: Double): ACContinuousActionAgent = {
    require(alpha >= 0)
    require(rewardDecay >= 0 && rewardDecay <= 1)
    require(valueDecay >= 0 && valueDecay <= 1)

    ACContinuousActionAgent(
      actor = actor,
      critic = critic,
      avg = Nd4j.create(Array(avg)),
      alpha = Nd4j.create(Array(alpha)),
      rewardDecay = Nd4j.create(Array(rewardDecay)),
      valueDecay = Nd4j.create(Array(valueDecay)))
  }

  /**
   * Returns mu, h, sigma
   *
   * @param out the output of actor network
   */
  def muHSigma(out: INDArray): (INDArray, INDArray, INDArray) = {
    val mu = Utils.clip(out.getColumn(0), -MuRange, MuRange)
    val h = Utils.clip(out.getColumn(1), -HRange, HRange)
    val sigma = Transforms.exp(h)
    (mu, h, sigma)
  }

  /**
   * Returns the target mu, h
   *
   * @param action the action
   * @param alpha  the alpha parameter
   * @param delta  the TD Error
   * @param mu     the mu
   * @param h      the h sigma
   * @param sigma  the sigma
   */
  def computeActorTarget(action: INDArray,
                         alpha: INDArray,
                         delta: INDArray,
                         mu: INDArray,
                         h: INDArray,
                         sigma: INDArray): (INDArray, INDArray) = {
    val sigma2 = sigma.mul(sigma)
    val deltaAction = action.sub(mu)
    val ratio = deltaAction.div(sigma)
    // deltaMu = 2 (action - mu) / sigma^2 delta
    val deltaMu = deltaAction.div(sigma2).muli(delta).muli(2)
    // deltaH = (2 (action - mu)^2 / sigma^2) - 1) delta
    val deltaH = ratio.mul(ratio).muli(2).subi(1).muli(delta).muli(alpha)
    val mu1 = Utils.clip(mu.add(deltaMu), -MuRange, MuRange)
    val h1 = Utils.clip(h.add(deltaH), -HRange, HRange)
    (mu1, h1)
  }

  /**
   * Returns (delta, v', avg')
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
    // v0' = (v1 + r - avgR) vDecay + avgR (1 - vDecay)
    val target=v1.add(reward).subi(avg)
    val delta = target.sub(v0)
    val newv0 = target.mul(valueDecay).addi(avg.mul(valueDecay.sub(1).negi()))
    // avgR' = avgR rDecay + r (1 - rDecay)
    val newAvg = rewardDecay.mul(avg).addi(rewardDecay.sub(1).negi.muli(reward))
    (delta, newv0, newAvg)
  }

}
