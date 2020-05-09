package org.mmarini.scalarl.v3.envs

import org.mmarini.scalarl.v3._
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.api.rng.Random
import org.nd4j.linalg.ops.transforms.Transforms

case class ContinuousActionEnv(x: INDArray, t: INDArray) extends Env {
  /** Returns the action configuration */
  override val actionConfig: Seq[ActionConfig] = Seq(ContinuousAction)
  val TilesEncoder: Tiles = Tiles(2L)

  /**
   * Computes the next status of environment executing an action.
   *
   * @param action the executing action
   * @param random the random generator
   * @return a n-uple with:
   *         - the environment in the next status,
   *         - the reward for the action,
   */
  override def change(action: INDArray, random: Random): (Env, INDArray) = {
    val clipAction = Utils.clip(action, -10, 10)
    val reward = Transforms.pow(clipAction.sub(x), 2).negi()
    (copy(t = t.add(1)), reward)
  }

  /** Returns the number of signals */
  override def signalsSize: Int = TilesEncoder.noFeatures.toInt

  /** Returns the [[Observation]] for the environment */
  override def observation: Observation = {
    val vx = x.sub(-1)
    val signals = Utils.features(TilesEncoder.features(vx), TilesEncoder.noFeatures)
    INDArrayObservation(
      signals = signals,
      time = t)
  }
}
