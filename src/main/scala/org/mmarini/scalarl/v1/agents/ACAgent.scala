package org.mmarini.scalarl.v1.agents

import java.io.File

import com.typesafe.scalalogging.LazyLogging
import io.circe.ACursor
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.util.ModelSerializer
import org.mmarini.scalarl.v1._
import org.nd4j.linalg.api.rng.Random
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.indexing.NDArrayIndex

/**
 * Actor critic agent
 *
 * @param actor       the actor network
 * @param critic      the critic network
 * @param actorRatio  the actor output ratio
 * @param criticRatio the critic output ratio
 * @param alpha       the alpha parameter
 * @param avg         the average rewards
 * @param beta        the beta parameter
 */
case class ACAgent(actor: MultiLayerNetwork,
                   critic: MultiLayerNetwork,
                   actorRatio: Double,
                   criticRatio: Double,
                   alpha: Double,
                   avg: Double,
                   beta: Double) extends Agent with LazyLogging {

  /**
   * Returns the new agent and the chosen action.
   * Chooses the action to be executed to the environment.
   *
   * @param observation the observation of environment
   * @param random      the random generator
   */
  override def chooseAction(observation: Observation, random: Random): Action = {
    val prefs = actor.output(observation.signals).mul(actorRatio)
    val pi = Utils.softMax(prefs, observation.actions)
    Utils.randomInt(pi)(random)
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

    // Critic update
    val criticLabel = Nd4j.create(Array(newv0))
    val newCritic = critic.clone()
    newCritic.fit(s0.signals, criticLabel.div(criticRatio))
    val score = delta * delta

    // Actor update
    val prefs = actor.output(s0.signals).mul(actorRatio)
    val pi = Utils.softMax(prefs, s0.actions)
    val actorLabel = prefs.dup().subi(pi)
    actorLabel.get(NDArrayIndex.point(action)).addi(1)
    actorLabel.muli(delta * alpha)
    val actorLabel1 = actorLabel.sub(actorLabel.mean())
    val newActor = actor.clone()
    newActor.fit(s0.signals, actorLabel1.div(actorRatio))
    logger.whenDebugEnabled {
      val v2 = v(newCritic, criticRatio, s0)
      val pi1 = Utils.softMax(actorLabel1, s0.actions)
      val pr2 = newActor.output(s0.signals).mul(actorRatio)
      val pi2 = Utils.softMax(pr2, s0.actions)
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
      logger.debug("  pr'     = {}", actorLabel)
      logger.debug("  npr'    = {}", actorLabel1)
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
  private def v(obs: Observation): Double = v(critic, criticRatio, obs)

  /**
   * Returns the value of a state
   *
   * @param critic      the critic
   * @param criticRatio the ratio of critic
   * @param obs         the observation
   */
  private def v(critic: MultiLayerNetwork, criticRatio: Double, obs: Observation): Double = if (obs.endUp) {
    0
  } else {
    critic.output(obs.signals).getDouble(0L) * criticRatio
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
   * @param file the model folder
   * @return the agents
   */
  override def writeModel(file: String): Agent = {
    ModelSerializer.writeModel(actor, new File(file, "actor.zip"), false)
    ModelSerializer.writeModel(critic, new File(file, "critic.zip"), false)
    this
  }
}
