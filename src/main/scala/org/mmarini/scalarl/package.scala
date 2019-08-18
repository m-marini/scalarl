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

package org.mmarini

import org.nd4j.linalg.api.ndarray.INDArray

/**
 * Provides type definitions for common data structures.
 *
 */
package object scalarl {
  /** Action performed by agent to change status in the environment */
  type Action = Int

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
   * Defines for each channel the number of values
   * 
   * {{{ Array(1, 2, 3) }}}
   * identifies 3 channels:
   *   - the first channel has just 1 value,
   *   - the second channel has 2 values
   *   - the third channel has 3 values.
   *  
   * The corresponding policy is a vector of 6 values:
   * 
   *  - `policy[0]` is the action value for the the only value of first channel
   *  - `policy[1]` is the action value for the the first value of second channel
   *  - `policy[2]` is the action value for the the second value of second channel
   *  - `policy[3]` is the action value for the the first value of third channel
   *  - `policy[4]` is the action value for the the second value of third channel
   *  - `policy[5]` is the action value for the the third value of third channel
   */
  type ActionChannelConfig = Array[Int]
  
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
