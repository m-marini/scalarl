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
import io.circe.ACursor
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.util.ModelSerializer
import org.mmarini.scalarl.v2._
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
 * @param noActions            number of actions
 * @param avgReward            the average reward
 * @param epsilon              epsilon greedy parameter
 * @param kappa                advance residual parameter
 * @param beta                 average reward step parameter
 * @param model                dyna+ model
 * @param maxModelSize         dyna+ model size
 * @param minModelSize         dyna+ minimum model size to plan
 * @param planningStepsCounter dyna+ model step counter
 * @param kappaPlus            dyna+ model kappa parameter
 * @param tolerance            dyna+ model status tollerance
 */
case class ExpSarsaAgent(net: MultiLayerNetwork,
                         noActions: Int,
                         avgReward: Double,
                         epsilon: Double,
                         kappa: Double,
                         beta: Double,
                         model: Seq[Feedback],
                         maxModelSize: Int,
                         minModelSize: Int,
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
  override def chooseAction(observation: Observation, random: Random): Action = {
    val q0 = Utils.indexed(q(observation), observation.actions)
    val pr = Utils.egreedy(q0, epsilon)
    val action = Utils.randomInt(pr)(random)
    action
  }

  /**
   * Returns the policy for an observation
   *
   * @param observation the observation
   */
  def q(observation: Observation): Policy = net.output(observation.signals)

  /**
   * Returns the fit agent by optimizing its strategy policy and the score
   *
   * @param feedback the feedback from the last step
   * @param random   the random generator
   */
  override def fit(feedback: Feedback, random: Random): (Agent, Double) = {
    val newModel = learnModel(feedback)
    val newNet = net.clone()
    val (newAvg, score) = train(newNet, feedback, avgReward)
    //val postLearnt = newNet.output(feedback.s0.signals)
    val newAvg1 = plan(newNet, newModel, feedback.s1.time, newAvg, random)
    //val postPlan = newNet.output(feedback.s0.signals)
    val newAgent = copy(net = newNet, model = newModel, avgReward = newAvg1)
    (newAgent, score)
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
  private def createData(feedback: Feedback, avg: Double): (INDArray, INDArray, Double, Double) = feedback match {
    case Feedback(obs0, action, reward, obs1) =>
      // Computes state values
      val q0 = q(obs0)
      val v0 = Utils.v(q0, action)

      val q1 = Utils.indexed(q(obs1), obs1.actions)
      val v1 = Utils.vExp(q1, Utils.egreedy(q1, epsilon))

      // Compute new q0 = v1 - Rm + R and delta = v1 - Rm + R - v0
      val newv0 = v1 + reward - avg
      val delta = newv0 - v0
      val score = delta * delta

      // Update average rewards
      val newAvg = avg + delta * beta

      // Computes labels
      val labels = q0.dup()
      labels.putScalar(action, newv0)
      //            logger.debug("---------------------------------------------------------------")
      //            logger.debug("  s0     = {}", obs0.signals)
      //            logger.debug("  action = {}", action)
      //            logger.debug("  reward = {}", reward)
      //            logger.debug("  s1     = {}", obs1.signals)
      //            logger.debug("  q0     = {} -> q0'     = {}", q0, labels)
      //            logger.debug("  q1     = {}", q1)
      //            logger.debug("  v0     = {} -> v0'     = {}", v0, newv0)
      //            logger.debug("  v1     = {}", v1)
      //            logger.debug("  avg(R) = {} -> avg'(R) = {}", avg, newAvg)
      //            logger.debug("  delta  = {}", delta)
      (obs0.signals, labels, newAvg, score)
  }

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
                   avg: Double,
                   random: Random): Double = {
    if (model.size > minModelSize) {
      var avg1 = avg
      for {_ <- 1 to planningStepsCounter} {
        val idx = random.nextInt(model.size)
        val feedback = model(idx)
        // compute reward bonus for late transitions
        val dt = time - feedback.s1.time
        val bouns = kappaPlus * Math.sqrt(dt)
        // Create inputs
        val (newAvg, _) = train(net, feedback, avg1)
        avg1 = newAvg
      }
      avg1
    } else {
      avg
    }
  }

  /**
   * Returns the new average reward and the score by training from feedback
   *
   * @param net      the neural network
   * @param feedback the feedback
   * @param avg      the average reward
   */
  private def train(net: MultiLayerNetwork, feedback: Feedback, avg: Double): (Double, Double) = {
    val (inputs, labels, newAvg, score) = createData(feedback, avg)
    // Train network
    net.fit(inputs, labels)
    (newAvg, score)
  }
}

/** The factory of [[ExpSarsaAgent]] */
object ExpSarsaAgent {

  /**
   * Returns the Dyna+Agent
   *
   * @param conf      the configuration element
   * @param noInputs  the neural inputs
   * @param noActions the number of action
   */
  def fromJson(conf: ACursor)(noInputs: Int, noActions: Int): ExpSarsaAgent = {
    val tolerance = loadTolerance(conf.downField("tolerance"))
    val avgReward = conf.get[Double]("avgReward").right.get
    val net = AgentNetworkBuilder.fromJson(conf.downField("network"))(noInputs, noActions)
    ExpSarsaAgent(net = net,
      noActions = noActions,
      model = Seq(),
      beta = conf.get[Double]("beta").right.get,
      avgReward = avgReward,
      maxModelSize = conf.get[Int]("maxModelSize").right.get,
      minModelSize = conf.get[Int]("minModelSize").right.get,
      epsilon = conf.get[Double]("epsilon").right.get,
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

  private case class Interval(interval: (Int, Int), value: Double) {
    require(interval._1 >= 0)
    require(interval._2 >= interval._1)
  }

}