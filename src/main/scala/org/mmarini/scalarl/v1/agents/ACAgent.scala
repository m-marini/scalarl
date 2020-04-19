package org.mmarini.scalarl.v1.agents

import com.typesafe.scalalogging.LazyLogging
import io.circe.ACursor
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.mmarini.scalarl.v1._
import org.nd4j.linalg.api.rng.Random
import org.nd4j.linalg.factory.Nd4j

/**
 * Actor critic agent
 *
 * @param actor  the actor network
 * @param critic the critic network
 * @param avg    the average rewards
 * @param beta   the beta parameter
 * @param beta1  the beta1 parameter
 */
case class ACAgent(actor: MultiLayerNetwork,
                   critic: MultiLayerNetwork,
                   avg: Double,
                   beta: Double,
                   beta1: Double) extends Agent with LazyLogging {

  /**
   * Returns the new agent and the chosen action.
   * Chooses the action to be executed to the environment.
   *
   * @param observation the observation of environment
   * @param random      the random generator
   */
  override def chooseAction(observation: Observation, random: Random): Action =
    Utils.randomInt(pi(observation))(random)

  /**
   * Returns the policy for a given status
   *
   * @param obs the status observation
   */
  private def pi(obs: Observation): Policy = {
    val prefs = actor.output(obs.signals)
    val pr = Utils.softMax(prefs, obs.actions)
    pr
  }

  /**
   * Returns the fit agent by optimizing its strategy policy and the score
   *
   * @param feedback the feedback from the last step
   * @param random   the random generator
   */
  override def fit(feedback: Feedback, random: Random): (Agent, Double) = {
    val Feedback(s0, action, reward, s1) = feedback
    val v0 = v(s0)
    val v1 = v(s1)
    val newv0 = v1 + reward - avg
    val delta = newv0 - v0
    val newAvg = avg + delta * beta
    val criticLabel = Nd4j.create(Array(newv0))
    val newCritic = critic.clone()
    newCritic.fit(s0.signals, criticLabel)
    val score = delta * delta

    // Simplified actor update
    val pr = actor.output(s0.signals).dup()
    pr.putScalar(action, pr.getDouble(action.toLong) + delta * beta1)
    val newActor = actor.clone()
    newActor.fit(s0.signals, pr)
    val newAgent = copy(actor = newActor, critic = newCritic, avg = newAvg)
    (newAgent, score)
  }

  /**
   * Returns the value of a state
   *
   * @param obs the observation
   */
  private def v(obs: Observation): Double = if (obs.endUp) {
    0
  } else {
    critic.output(obs.signals).getDouble(0L)
  }

  /**
   * Returns the score for a feedback
   *
   * @param feedback the feedback from the last step
   */
  override def score(feedback: Feedback): Double = {
    val Feedback(s0, action, reward, s1) = feedback
    val v0 = v(s0)
    val v1 = v(s1)
    val delta = v1 + reward - avg - v0
    val score = delta * delta
    score
  }

  /**
   * Returns the reset agent
   *
   * @param random the random generator
   */
  override def reset(random: Random): Agent = this

  /**
   * Writes the agent status to file
   *
   * @param file the filename
   * @return the agents
   */
  override def writeModel(file: String): Agent = ???
}

object ACAgent {
  /**
   * Returns the Dyna+Agent
   *
   * @param conf   the configuration element
   * @param actor  the neural network of actor
   * @param critic the neural network of critic
   */
  def apply(conf: ACursor, actor: MultiLayerNetwork, critic: MultiLayerNetwork): ACAgent = ACAgent(
    actor = actor,
    critic = critic,
    beta = conf.get[Double]("beta").right.get,
    beta1 = conf.get[Double]("beta1").right.get,
    avg = conf.get[Double]("avgReward").right.get
  )
}