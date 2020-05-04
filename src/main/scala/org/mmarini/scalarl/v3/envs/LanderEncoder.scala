package org.mmarini.scalarl.v3.envs

import org.nd4j.linalg.api.ndarray.INDArray

/** Encodes the status */
trait LanderEncoder {
  /**
   * Returns the input signals
   *
   * @param status the status
   */
  def signals(status: LanderStatus): INDArray

  /** Returns the number of signals */
  def noSignals: Int
}
