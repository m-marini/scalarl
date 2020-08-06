package org.mmarini.scalarl.v4.envs

import org.nd4j.linalg.api.ndarray.INDArray

/** Encodes the status */
trait LanderEncoder {
  /**
   * Returns the input signals
   *
   * @param s the signals
   */
  def signals(s: INDArray): INDArray

  /** Returns the number of signals */
  def noSignals: Int
}
