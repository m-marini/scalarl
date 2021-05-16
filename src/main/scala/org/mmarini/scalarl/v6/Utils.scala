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

package org.mmarini.scalarl.v6

import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.api.rng.Random
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.factory.Nd4j._
import org.nd4j.linalg.indexing.NDArrayIndex
import org.nd4j.linalg.ops.transforms.Transforms
import org.nd4j.linalg.ops.transforms.Transforms._

object Utils {

  /**
   * Returns the function that clip, denormalize and center the values
   *
   * @param range the output range
   */
  def clipDenormalizeAndCenter(range: INDArray): INDArray => INDArray = {
    val clip = clipAndDenormalize(range)
    val result = (x: INDArray) => {
      val pr = clip(x)
      val mean = pr.mean()
      val pr1 = pr.subi(mean)
      pr1
    }
    result
  }

  /**
   * Returns the denormalize function of row vector after clipping
   * The function returns ranges for (-1, 1) ranges
   *
   * @param ranges the range of transformation the first row contains minimum values
   *               and the second row contains the maximum values
   */
  def clipAndDenormalize(ranges: INDArray): INDArray => INDArray = {
    val fromRanges = vstack(ones(1, ranges.size(1)).negi(), ones(1, ranges.size(1)))
    clipAndTransform(fromRanges, ranges)
  }

  /**
   * Returns the action output value from action index
   *
   * @param noValues numbers of values
   * @param range    output range
   */
  def encode(noValues: Int, range: INDArray): Int => INDArray = {
    val min = range.getDouble(0L)
    val scale = (range.getDouble(1L) - min) / (noValues - 1)
    val actions = for {
      i <- 0 until noValues
    } yield {
      ones(1).muli(i * scale + min)
    }
    val result = (action: Int) => actions(Math.min(Math.max(0, action), noValues - 1))
    result
  }

  /**
   * Returns the action features from action output value
   *
   * @param noValues numbers of values
   * @param range    output range
   */
  def decode(noValues: Int, range: INDArray): INDArray => INDArray = {
    val min = range.getDouble(0L)
    val scale = (noValues - 1) / (range.getDouble(1L) - min)
    val fClip = clip(range)
    val result = (action: INDArray) => {
      val fc = fClip(action)
      val actionIndex = round(fc.subi(min).muli(scale)).getInt(0)
      val features = zeros(noValues)
      features.getScalar(actionIndex.toLong).assign(1)
      features
    }
    result
  }

  /**
   * Returns the features vector with ones at indices
   *
   * @param indices indices to set ones
   * @param size    size of result
   */
  def features(indices: Seq[Long], size: Long): INDArray = {
    require(size > 0)
    indices.foreach(i => require(i >= 0 && i < size, s"indices=$indices size=$size"))
    val result = zeros(1, size)
    indices.foreach(result.putScalar(_, 1))
    result
  }

  /**
   * Returns random integer for a distribution function
   *
   * @param x the distribution function
   */
  def randomInt(x: INDArray): Random => Int = cdfRandomInt(cdf(x))

  /**
   * Returns the cumulative distribution function for the actions
   *
   * @param x the distribution function
   */
  def cdf(x: INDArray): INDArray = {
    val n = x.length()
    val cdf = x.dup()
    for (i <- 1L until n) {
      val v = cdf.getDouble(i - 1) + cdf.getDouble(i)
      cdf.putScalar(i, v)
    }
    cdf
  }

  /**
   * Returns random integer for a cdf
   *
   * @param x      the cumulative distribution function
   * @param random the random generator
   */
  def cdfRandomInt(x: INDArray)(random: Random): Int = {
    val n = x.length()
    val seq = for {
      i <- 0L until n
    } yield x.getDouble(i)
    val y = random.nextDouble()
    val result = seq.indexWhere(y < _)
    require(result >= 0, s"$x")
    result
  }

  /**
   *
   */

  /**
   * Returns the values of x clipped by scaling to desired max value
   *
   * @param x        the values
   * @param maxValue the max value
   */
  def scaleClip(x: INDArray, maxValue: Double): INDArray = {
    val max = abs(x).max().getDouble(0L)
    val result = if (max > maxValue) {
      x.mul(maxValue / max)
    } else {
      x
    }
    result
  }

  /**
   * Returns the normalizer function of row vector after clipping
   * The function returns ranges -1, 1 for defined ranges
   *
   * @param ranges the range of transformation the first row contains minimum values
   *               and the second row contains the maximum values
   */
  def clipAndNormalize01(ranges: INDArray): INDArray => INDArray = {
    val toRanges = vstack(zeros(1, ranges.size(1)), ones(1, ranges.size(1)))
    clipAndTransform(ranges, toRanges)
  }

  /**
   * Returns the normalizer function of row vector after clipping
   * The function returns ranges -1, 1 for defined ranges
   *
   * @param ranges the range of transformation the first row contains minimum values
   *               and the second row contains the maximum values
   */
  def clipAndNormalize(ranges: INDArray): INDArray => INDArray = {
    val toRanges = vstack(ones(1, ranges.size(1)).negi(), ones(1, ranges.size(1)))
    clipAndTransform(ranges, toRanges)
  }

  /**
   * Returns the normalizer function of row vector after clipping
   * The function returns ranges -1, 1 for defined ranges
   *
   * @param fromRanges the range of inputs the first row contains minimum values
   *                   and the second row contains the maximum values
   * @param toRanges   the range of outputs the first row contains minimum values
   *                   and the second row contains the maximum values
   */
  def clipAndTransform(fromRanges: INDArray, toRanges: INDArray): INDArray => INDArray = {
    val m = toRanges.getRow(1).sub(toRanges.getRow(0)).divi(fromRanges.getRow(1).sub(fromRanges.getRow(0)))
    val q = m.mul(fromRanges.getRow(0)).subi(toRanges.getRow(0)).negi()
    val cl = clip(fromRanges)
    x => cl(x).muli(m).addi(q)
  }

  /**
   * Returns the clip function
   *
   * @param range range values row(0) = min, row(1) = max
   */
  def clip(range: INDArray): INDArray => INDArray = (x: INDArray) =>
    Transforms.min(Transforms.max(x, range.getRow(0), true), range.getRow(1), false)

  /**
   * Returns the distribution function for linear preferences
   *
   * @param x the preferences
   */
  def df(x: INDArray): INDArray = x.div(x.sumNumber().doubleValue())

  /**
   * Returns the e-greedy policy for the given action values
   *
   * @param q       actions values
   * @param epsilon epsilon
   */
  def eGreedy(q: INDArray, epsilon: INDArray): INDArray = {
    val n = q.length()
    if (n > 1) {
      val act = q.argMax().getInt(0)
      val result = Nd4j.ones(1, n).muli(epsilon).divi(n - 1)
      result.put(act, epsilon.sub(1).negi())
      result
    } else {
      Nd4j.ones(1)
    }
  }

  /**
   * Returns the indices for not zero values
   *
   * @param x the values
   */
  def find(x: INDArray): Seq[Long] = for {
    i <- 0L until x.length()
    if x.getDouble(i) != 0.0
  } yield i

  /**
   * Returns the indexed values
   *
   * @param x       the values
   * @param indices the indices
   */
  def indexed(x: INDArray, indices: Seq[Long]): INDArray = {
    val idx = NDArrayIndex.indices(indices.toArray: _*)
    x.get(NDArrayIndex.all(), idx)
  }

  /**
   * Returns the status value for the action and argMax
   * The status value of a policy is the max value for the channel
   *
   * @param q      action value
   * @param action action
   * @return the array of values for each channel and the indices of max action features
   */
  def v(q: INDArray, action: Int): INDArray = q.getColumn(action)

  /**
   * Returns the expected status value
   *
   * @param q  the action values
   * @param pi the policy
   */
  def vExp(q: INDArray, pi: INDArray): INDArray = q.mul(pi).sum()

  /**
   * Returns the greedy status value
   *
   * @param q the actions values
   */
  def vStar(q: INDArray): INDArray = q.max()
}
