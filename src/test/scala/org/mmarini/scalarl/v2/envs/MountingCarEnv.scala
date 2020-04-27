package org.mmarini.scalarl.v2.envs

import org.mmarini.scalarl.v2._
import org.mmarini.scalarl.v2.envs.MountingCarEnv._
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.api.rng.Random
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.ops.transforms.Transforms._

case class MountingCarEnv(x: INDArray,
                          v: INDArray,
                          t: INDArray) extends EnvContinuousAction {
  val TilesEncoder = Tiles(1L, 1L)

  /**
   * Computes the next status of environment executing an action.
   *
   * @param action the executing action
   * @param random the random generator
   * @return a n-uple with:
   *         - the environment in the next status,
   *         - the reward for the action,
   */
  override def change(action: INDArray, random: Random): (EnvContinuousAction, INDArray) = {
    if (x.getDouble(0L) >= XRight) {
      (initial(random, t), Nd4j.zeros(1))
    } else {
      val clipAction = Utils.clip(action, -1, 1)
      // v' = v + 1e-3 action - 2.5e-3 cos(3 x)
      val v1 = v.add(clipAction.mul(1e-3)).subi(cos(x.mul(3)).muli(2.5e-3))
      val v1Clip = Utils.clip(v1, VMin, VMax)
      // x' = x + v'
      val x1 = x.add(v1Clip)
      val x1Clip = Utils.clip(x1, XLeft, XRight)
      val v2 = if (x1Clip.getDouble(0L) > XLeft) v1Clip else Nd4j.zeros(1)
      val reward = if (x1Clip.getDouble(0L) >= XRight) Nd4j.ones(1) else Nd4j.ones(1).negi()
      (copy(x = x1Clip, v = v2, t = t.add(1)), reward)
    }
  }

  /** Returns the number of signals */
  override def signalsSize: Int = TilesEncoder.noFeatures.toInt

  /** Returns the [[Observation]] for the environment */
  override def observation: Observation = {
    val vx = x.sub(XLeft).divi(XRight - XLeft)
    val vv = v.sub(VMin).divi(VMax - VMin)
    val point = Nd4j.hstack(vx, vv)

    val signals = Utils.features(TilesEncoder.features(point), TilesEncoder.noFeatures)
    INDArrayObservation(
      signals = signals,
      time = t)
  }
}

object MountingCarEnv {
  val XLeft = -1.2
  val XRight = -0.5
  val VMin = -0.07
  val VMax = 0.07
  val X0Min = -0.6
  val X0Max = -0.4

  /**
   *
   * @param random
   */
  def initial(random: Random, t: INDArray): MountingCarEnv = {
    val x0 = Nd4j.create(Array(random.nextDouble() * (XRight - XLeft) - XLeft))
    MountingCarEnv(x = x0, v = Nd4j.zeros(1), t = t)
  }

  /**
   *
   * @param random
   */
  def initial(random: Random): MountingCarEnv = initial(random, Nd4j.zeros(1))
}