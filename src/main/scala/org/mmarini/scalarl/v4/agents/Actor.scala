package org.mmarini.scalarl.v4.agents

import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.api.rng.Random

trait Actor {

  /** Returns the number of outputs */
  def noOutputs: Int

  /**
   * Returns the actor labels
   *
   * @param outputs the outputs
   * @param actions the actions
   * @param delta   the td error
   */
  def computeLabels(outputs: Array[INDArray], actions: INDArray, delta: INDArray): Map[String, Any]

  /**
   * Returns the action choosen by the actor
   *
   * @param outputs the network outputs
   * @param random  the random generator
   */
  def chooseAction(outputs: Array[INDArray], random: Random): INDArray

  /** Returns the dimension index of action agent */
  def dimension: Int
}
