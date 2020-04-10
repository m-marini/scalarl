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

package org.mmarini.scalarl.ts.agents

import java.io.File

import com.typesafe.scalalogging.LazyLogging
import io.circe.ACursor
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.util.ModelSerializer
import org.mmarini.scalarl.ts._
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.api.rng.Random
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.indexing.NDArrayIndex
import org.nd4j.linalg.ops.transforms.Transforms._

/**
 * The agent acting in the environment by QLearning T(0) algorithm.
 *
 * Generates actions to change the status of environment basing on observation of the environment
 * and the internal strategy policy.
 *
 * Updates its strategy policy to optimize the return value (discount sum of rewards)
 * and the observation of resulting environment
 *
 * @param net                  the neural network
 * @param config               the action configuration
 * @param avgReward            the average reward
 * @param epsilon              epsilon greedy parameter
 * @param gamma                discount reward
 * @param kappa                advance residual parameter
 * @param beta                 average reward step parameter
 * @param model                dyna+ model
 * @param maxModelSize         dyna+ model size
 * @param planningStepsCounter dyna+ model step counter
 * @param kappaPlus            dyna+ model kappa parameter
 * @param tolerance            dyna+ model status tollerance
 */
case class DynaQPlusAgent(net: MultiLayerNetwork,
                          config: DiscreteActionChannels,
                          avgReward: INDArray,
                          epsilon: Double,
                          gamma: Double,
                          kappa: Double,
                          beta: Double,
                          model: Seq[Feedback],
                          maxModelSize: Int,
                          planningStepsCounter: Int,
                          kappaPlus: Double,
                          tolerance: Option[INDArray]) extends Agent with LazyLogging {
  /**
   * Returns the new agent and the chosen action.
   * Chooses the action to be executed to the environment.
   *
   * @param observation the observation of environment
   * @param random      the random generator
   */
  override def chooseAction(observation: Observation, random: Random): ChannelAction = {
    val valueMask = observation.actions
    val action = if (random.nextDouble() < epsilon) {
      // Explore action
      config.random(valueMask, random)
    } else {
      // Exploit action with greedy policy
      greedyAction(observation)
    }
    action
  }

  /**
   * Returns the best action estimate by network for a give observation
   *
   * @param observation the observation
   */
  private def greedyAction(observation: Observation): ChannelAction = {
    val p = policy(observation)
    val mask = observation.actions
    config.greedyAction(p, mask)
  }

  /**
   * Returns the fit agent by optimizing its strategy policy and the score
   *
   * @param feedback the feedback from the last step
   * @param random   the random generator
   */
  override def fit(feedback: Feedback, random: Random): (Agent, Double) = {
    val newModel = learnModel(feedback)
    val (inputs, labels, newAvg, score) = createData(feedback, avgReward)
    val newNet = net.clone()
    newNet.fit(inputs, labels)
    val (learntNet, newAvg1) = plan(newNet, newModel, feedback.s1.time, newAvg, random)
    val newAgent = copy(net = learntNet, model = newModel, avgReward = newAvg1)
    // logger.debug("  learnt = {}", newAgent.policy(feedback.s0))
    (newAgent, score)
  }

  /**
   * Returns the model with a new feedback
   *
   * @param feedback the feedback
   */
  private def learnModel(feedback: Feedback): Seq[Feedback] = {
    if (maxModelSize > 0) {
      val removed = model.filterNot(f => {
        tolerance.map(t => {
          if (f.action == feedback.action) {
            val d = f.s0.signals.sub(feedback.s0.signals)
            val diff = abs(d).subi(t)
            // array with ones if diff > 0 (states are different)
            val neqf = diff.gti(0)
            // count differences
            val diffCount = neqf.sumNumber().doubleValue()
            val eq = diffCount == 0
            eq
          } else {
            false
          }
        }).getOrElse(f.s0.signals == feedback.s0.signals && f.action == feedback.action)
      })

      val cap = if (removed.size >= maxModelSize) removed.tail else removed
      cap :+ feedback
    } else {
      model
    }
  }

  /**
   * Returns the trained network by planning with the model and the average rewards.
   *
   * The implementation changes the input network as side effect of learning process.
   *
   * @param net    the network
   * @param model  the environment model
   * @param time   the instant of planning used to compute the reward bouns for late transitions
   * @param avg    the average rewards
   * @param random the random generator
   */
  private def plan(net: MultiLayerNetwork,
                   model: Seq[Feedback],
                   time: Double,
                   avg: INDArray,
                   random: Random): (MultiLayerNetwork, INDArray) = {
    var avg1 = avg
    for {_ <- 1 to planningStepsCounter} {
      val idx = random.nextInt(model.size)
      val feedback = model(idx)
      // compute reward bonus for late transitions
      val dt = time - feedback.s1.time
      val bouns = kappaPlus * Math.sqrt(dt)
      // Create inputs
      val feedback1 = feedback.copy(reward = feedback.reward + bouns)
      val (inputs, labels, newAvg, _) = createData(feedback1, avg1)
      avg1 = newAvg
      // Train network
      net.fit(inputs, labels)
    }
    (net, avg1)
  }

  /**
   * Returns the score a feedback
   *
   * @param feedback the feedback from the last step
   */
  override def score(feedback: Feedback): Double =
    createData(feedback, avgReward)._4

  /**
   * Returns the 3-upla with data for fit
   *
   * @param feedback the feedback
   * @param avg      the average rewards
   * @return the input signals, the output label, the new advantage reward, the score
   */
  private def createData(feedback: Feedback, avg: INDArray): (INDArray, INDArray, INDArray, Double) = feedback match {
    case Feedback(obs0, action, reward, obs1) =>
      val q0 = policy(obs0)
      val q1 = policy(obs1)
      // Computes state values
      val (v1, _) = config.statusValue(q1, obs1.actions)
      val v0 = config.actionValues(q0, action)

      // Compute new q0 = v1 - Rm + R and delta = v1 - Rm + R - v0
      val newQ0 = v1.sub(avg).addi(reward)
      val delta = newQ0.sub(v0)
      val score = delta.sumNumber().doubleValue()

      // Update average rewards
      val newAvg = avg.add(delta.mul(beta))
      val idx = config.notZeroIndices(action)
      // Computes labels
      val labels = q0.dup()
      for {(i, j) <- idx.zipWithIndex} {
        labels.putScalar(i, newQ0.getDouble(j.toLong))
      }
      //      logger.debug("---------------------------------------------------------------")
      //      logger.debug("  s0     = {}", obs0.signals)
      //      logger.debug("  action = {}", action)
      //      logger.debug("  reward = {}", reward)
      //      logger.debug("  s1     = {}", obs1.signals)
      //      logger.debug("  q0     = {}", q0)
      //      logger.debug("  q1     = {}", q1)
      //      logger.debug("  v0     = {}", v0)
      //      logger.debug("  v1     = {}", v1)
      //      logger.debug("  avg(R) = {}", avg)
      //      logger.debug("  delta  = {}", delta)
      //      logger.debug("  q0'    = {}", newQ0)
      //      logger.debug("  labels = {}", labels)
      (obs0.signals, labels, newAvg, score)
  }

  /**
   * Returns the policy for an observation
   *
   * @param observation the observation
   */
  def policy(observation: Observation): Policy = if (observation.endUp) {
    config.zeroPolicy
  } else {
    net.output(observation.signals)
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
  override def writeModel(file: String): Agent = {
    ModelSerializer.writeModel(net, new File(file), false)
    this
  }
}

private case class Interval(interval: (Int, Int), value: Double) {
  require(interval._1 >= 0)
  require(interval._2 >= interval._1)
}

/** The factory of [[DynaQPlusAgent]] */
object DynaQPlusAgent {

  /**
   * Returns the Dyna+Agent
   *
   * @param conf   the configuration element
   * @param net    the neural network
   * @param config the action channel configuration
   */
  def apply(conf: ACursor, net: MultiLayerNetwork, config: DiscreteActionChannels): DynaQPlusAgent = {
    val tolerance = loadTolerance(conf.downField("tolerance"))
    val avgReward = Nd4j.ones(config.noChannels).mul(conf.get[Double]("avgReward").right.get)
    DynaQPlusAgent(net = net,
      model = Seq(),
      config = config,
      beta = conf.get[Double]("beta").right.get,
      avgReward = avgReward,
      maxModelSize = conf.get[Int]("maxModelSize").right.get,
      epsilon = conf.get[Double]("epsilon").right.get,
      gamma = conf.get[Double]("gamma").right.get,
      kappa = conf.get[Double]("kappa").right.get,
      kappaPlus = conf.get[Double]("kappaPlus").right.get,
      planningStepsCounter = conf.get[Int]("planningStepsCounter").right.get,
      tolerance = tolerance)
  }

  /**
   * Returns the tolerance for the model
   *
   * @param cursor the tolerance configuration
   */
  private def loadTolerance(cursor: ACursor): Option[INDArray] = {
    val intervals = cursor.values.map(list => {
      list.map(item => {
        val interval = item.hcursor.get[List[Int]]("interval").right.get
        if (interval.size != 2) {
          throw new IllegalArgumentException("Wrong interval")
        }
        val value = item.hcursor.get[Double]("value").right.get
        Interval((interval.head, interval(1)), value)
      })
    })
    val result = intervals.map(inters => {
      val last = inters.map(_.interval._2).max
      val tolerance = Nd4j.zeros(last + 1L)
      inters.foreach(interval => {
        tolerance.get(NDArrayIndex.interval(
          interval.interval._1,
          interval.interval._2 + 1)).
          assign(interval.value)
      })
      tolerance
    })
    result
  }
}