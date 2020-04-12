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
   * Returns the state value for each channel
   *
   * @param q    the action values
   * @param mask the actions mask
   */
  def v(q: INDArray, mask: INDArray): INDArray = {
    val result = Nd4j.zeros(noChannels)
    for {(chInt, i) <- intervals.zipWithIndex} {
      val qch = q.get(chInt)
      val idx = Utils.find(mask.get(chInt))
      val value = Utils.indexed(qch, idx).maxNumber().doubleValue()
      result.putScalar(i, value)
    }
    result
  }

  /**
   * Returns the expected value
   *
   * @param q       the action values
   * @param mask    the available actions mask
   * @param epsilon epsilon parameter
   */
  def vExp(q: INDArray, mask: INDArray, epsilon: Double): INDArray = {
    val result = Nd4j.zeros(noChannels)
    for {(chInt, i) <- intervals.zipWithIndex} {
      val qch = q.get(chInt)
      val idx = Utils.find(mask.get(chInt))
      val qa = Utils.indexed(qch, idx)
      val pi = Utils.egreedy(qa, epsilon)
      val value = pi.mul(qa).sumNumber().doubleValue()
      result.putScalar(i, value)
    }
    result
  }

  /**
   * Returns the action indices of each channel
   *
   * @param data the action data value 1 for each action channel features
   */
  def actions(data: INDArray): Seq[Long] = {
    require(data.length() == size, s"action length [${data.length()}] should be $size")
    val idx = Utils.find(data)
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
    val maskIdx = Utils.find(action)
    Utils.indexed(policy, maskIdx)
  }

  /**
   * Returns the probabilities of actions selection for each channel
   *
   * @param q       the action values
   * @param mask    the mask of valid actions
   * @param epsilon the epsilon parameter
   */
  def egreedyPolicy(q: INDArray, mask: INDArray, epsilon: Double): INDArray = {
    val result = Nd4j.zeros(q.shape(): _*)
    val x = for {
      chInt <- intervals
    } yield {
      val chQ = q.get(chInt)
      val idx = Utils.find(mask.get(chInt))
      val m = Utils.indexed(chQ, idx)
      val pr = Utils.egreedy(m, epsilon)
      val res = result.get(chInt)
      for {(i, j) <- idx.zipWithIndex} {
        res.putScalar(i, pr.getDouble(j.toLong))
      }
    }
    result
  }

  /**
   * Returns a random action with a given policy
   *
   * @param p      the policy of action
   * @param random the random generator
   */
  def actionFromPolicy(p: INDArray)(random: Random): INDArray = {
    val actions = for {chInt <- intervals} yield
      Utils.randomInt(p.get(chInt))(random)
    action(actions: _*)
  }
}
