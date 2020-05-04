package org.mmarini.scalarl.v3.agents

import java.io.File

import org.mmarini.scalarl.v3.{Feedback, Observation}
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.api.rng.Random

trait Actor {
  /** Returns the dimension index of action agent */
  def dimension: Int

  /**
   * Returns chosen action.
   * Chooses the action to be executed to the environment.
   *
   * @param observation the observation of environment
   * @param random      the random generator
   */
  def chooseAction(observation: Observation, random: Random): INDArray

  /**
   * Returns the changed actor
   * The lerning method is applied to create a new fit actor
   *
   * @param feedback the feedback
   * @param delta    the TD Error
   * @param random   the random generator
   */
  def fit(feedback: Feedback, delta: INDArray, random: Random): Actor

  /**
   * Writes the agent status to a path
   *
   * @param path the path to store the files of model
   * @return the agents
   */
  def writeModel(path: File): Actor
}
