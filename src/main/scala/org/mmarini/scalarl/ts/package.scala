package org.mmarini.scalarl

import org.nd4j.linalg.api.ndarray.INDArray

package object ts {
  /** Reward received by agent in response to status change */
  type Reward = Double

  /** Flag to signal end of episode */
  type EndUp = Boolean

  /**
   * I
   * s a vector with the action values,
   * each action channels is a set of index with the vector and the highest value
   * for the action channels identifies the best action value for the channel
   */
  type Policy = INDArray

  /**
   * The [[ActionMask]] identifies the valid action channel values
   *
   *  - `1` value indicates a valid action a value,
   *  - `0` value indicates an invalid action value
   */
  type ActionMask = INDArray

  /**
   * Defines which value of channel action the agent has chosen
   * The vector contains value `1` for the selected values
   *
   * Given the configuration
   * {{{
   *   config = Array(1, 2, 3)
   * }}}
   * the [[ChannelAction]]
   * {{{
   * 	action = Array(1, 0, 1, 0, 1, 0)
   * }}}
   * defines the first value of first channel, the second value of second channel
   * and the first value of third channel
   */
  type ChannelAction = INDArray

  /**
   * Defines the values of status for the selected value channel.
   * The vaues of status is the highest values of policy for each channel action.
   * Each values state is replicated for each action channel.
   *
   * Given the configuration and the policy
   * {{{
   *   config = Array(1, 2, 3)
   *   policy = Array( 1.0, 2.5, 2.0, 3.0, 4.5, 2.0)
   * }}}
   *
   * The resulting state values are
   * {{{
   *   stateVales = Array( 1.0, 2.5, 2.5, 4.5, 4.5, 4.5)
   * }}}
   */
  type StatusValues = INDArray
}
