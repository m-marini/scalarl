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

package org.mmarini.scalarl.ts

import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.api.rng.Random
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.indexing.{INDArrayIndex, NDArrayIndex}

/**
 *
 * @param sizes size of each action channel
 */
case class DiscreteActionChannels(sizes: Array[Int]) {
  /** Returns the indices of channels (start, end inclusive) */
  val indices: Array[(Int, Int)] = {
    val (indexes, _) = sizes.foldLeft((Array[(Int, Int)](), 0)) {
      case ((indexes, start), length) =>
        val next = start + length
        val newIndexes = indexes :+ (start, next - 1)
        (newIndexes, next)
    }
    indexes
  }

  /** Returns the interval of actions */
  val intervals: Array[INDArrayIndex] =
    indices.map(i => NDArrayIndex.interval(i._1, i._2 + 1))

  /** Returns the number of action signals */
  val size: Int = sizes.sum

  /** Returns the zero policy (end episode values) */
  val zeroPolicy: INDArray = Nd4j.zeros(size)

  /**
   * Returns a random action
   *
   * @param mask   the mask of available actions
   * @param random the random generator
   */
  def random(mask: INDArray, random: Random): INDArray = {
    val idx = for {
      inter <- intervals
    } yield {
      val chMask = mask.get(inter)
      val maskIdx = notZeroIndices(chMask)
      require(maskIdx.nonEmpty, "mask should not be empty")
      val idx = if (maskIdx.isEmpty) {
        0
      } else {
        maskIdx(random.nextInt(maskIdx.length))
      }
      idx
    }
    action(idx: _*)
  }

  /**
   * Returns the index of not zero values
   *
   * @param data the data
   */
  def notZeroIndices(data: INDArray): Seq[Int] = for {
    i <- 0 until data.length().toInt
    if data.getInt(i) != 0
  } yield i

  /**
   * Returns the action signal
   *
   * @param actions the action indices of each channel
   */
  def action(actions: Int*): INDArray = {
    require(actions.length == sizes.length)
    for {(a, size) <- actions.zip(sizes)} {
      require(a >= 0 && a < size)
    }
    val result = Nd4j.zeros(size)
    for {(slice, value) <- intervals.zip(actions)} {
      result.get(slice).putScalar(value, 1)
    }
    result
  }

  /**
   * Returns the action indices of each channel
   *
   * @param data the action data value 1 for each action channel features
   */
  def actions(data: INDArray): Seq[Int] = {
    require(data.length() == size, s"action length [${data.length()}] should be $size")
    val idx = notZeroIndices(data)
    require(idx.length == sizes.length, s"number of actions [${idx.length}] should be ${sizes.length}")
    for {(a, (from, to)) <- idx.zip(indices)} {
      require(a >= from && a <= to, s"action value $a should be between $from to $to")
    }
    for {(a, (from, _)) <- idx.zip(indices)} yield a - from
  }

  /** Returns the number of channels */
  def noChannels: Int = sizes.length

  /**
   * Returns the action value of the action
   *
   * @param policy the policy
   * @param action the mask of action channel features
   * @return a vector of values of policy where mask is not zero
   */
  def actionValues(policy: Policy, action: INDArray): INDArray = {
    val maskIdx = notZeroIndices(action)
    val values = maskIdx.map(i => policy.getDouble(i.toLong)).toArray
    Nd4j.create(values)
  }

  /**
   * Returns the mask of max feature values for channels
   *
   * @param policy the policy
   * @param mask   the mask of available actions
   */
  def greedyAction(policy: Policy, mask: INDArray): INDArray = {
    val (_, idx) = statusValue(policy, mask)
    action(idx: _*)
  }

  /**
   * Returns the status value for the action and argMax
   * The status value of a policy is the max value for the channel
   *
   * @param policy action value
   * @param action mask of available actions
   * @return the array of values for each channel and the indices of max action features
   */
  def statusValue(policy: Policy, action: INDArray): (INDArray, Array[Int]) = {
    val x = for {interval <- intervals} yield {
      val p = policy.get(interval)
      val maskIdx = notZeroIndices(action.get(interval))
      val values = maskIdx.map(i => p.getDouble(i.toLong))
      val max = values.max
      val idx = values.indexOf(max)
      (max, maskIdx(idx))
    }
    val (vals, idx) = x.unzip
    val values = Nd4j.create(vals)
    (values, idx)
  }
}
