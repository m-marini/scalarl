package org.mmarini.scalarl.v2

import org.nd4j.linalg.api.ndarray.INDArray

package object v1 {
  /** Reward received by agent in response to status change */
  type Reward = Double

  /** Flag to signal end of episode */
  type EndUp = Boolean

  /** Is a vector with the probaiblity of actions */
  type Policy = INDArray

  /** Is a vector with the action values */
  type QValues = INDArray

  /** The [[ActionMask]] identifies the valid actions */
  type ActionMask = Seq[Long]

  /**
   * The action selected
   */
  type Action = Int

  /** The value of status */
  type VValue = Double
}
