package org.mmarini.scalarl.v4

import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.api.rng.Random
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.factory.Nd4j._
import org.nd4j.linalg.indexing.NDArrayIndex
import org.nd4j.linalg.ops.transforms.Transforms
import org.nd4j.linalg.ops.transforms.Transforms._

object Utils {

  /**
   * Returns the clip values
   *
   * @param x    the values
   * @param xMin minimum value
   * @param xMax maximum values
   * @param copy true if return value is a new copy
   */
  def clip(x: INDArray, xMin: Double, xMax: Double, copy: Boolean = true): INDArray = Transforms.min(Transforms.max(x, xMin, copy), xMax, copy)

  /**
   * Returns the clip values
   *
   * @param x     the values
   * @param range range values row(0) = min, row(1) = max
   * @param copy  true if return value is a new copy
   */
  def clip(x: INDArray, range: INDArray, copy: Boolean): INDArray = Transforms.min(Transforms.max(x, range.getRow(0), copy), range.getRow(1), copy)

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
   * @param x the distributino function
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
   * Returns the values of x cliped by scaling to desired max value
   *
   * @param x        the values
   * @param maxValue the max value
   */
  def scaleClip(x: INDArray, maxValue: Double): INDArray = {
    val max = abs(x).max().getDouble(0L)
    val result = if (max > maxValue)
      x.mul(maxValue / max)
    else
      x
    result
  }

  /**
   * Returns the denormalizer function of row vector
   * The function returns defined ranges form ranges 0, 1
   *
   * @param ranges the range of transformation the first row contains minimum values
   *               and the second row contains the maximum values
   */
  def denormalize01(ranges: INDArray): INDArray => INDArray = {
    val fromRanges = vstack(zeros(1, ranges.size(1)), ones(1, ranges.size(1)))
    transform(fromRanges, ranges)
  }

  /**
   * Returns the denormalizer function of row vector
   * The function returns defined ranges form ranges -1, 1
   *
   * @param ranges the range of transformation the first row contains minimum values
   *               and the second row contains the maximum values
   */
  def denormalize(ranges: INDArray): INDArray => INDArray = {
    val min = ones(1, ranges.size(1)).negi()
    val max = ones(1, ranges.size(1))
    val fromRanges = vstack(min, max)
    transform(fromRanges, ranges)
  }

  /**
   * Returns the normalizer function of row vector
   * The function returns ranges -1, 1 for defined ranges
   *
   * @param ranges the range of transformation the first row contains minimum values
   *               and the second row contains the maximum values
   */
  def normalize(ranges: INDArray): INDArray => INDArray = {
    val toRanges = vstack(ones(1, ranges.size(1)).negi(), ones(1, ranges.size(1)))
    transform(ranges, toRanges)
  }

  /**
   * Returns the normalizer function of row vector
   * The function returns ranges 0, 1 for defined ranges
   *
   * @param ranges the range of transformation the first row contains minimum values
   *               and the second row contains the maximum values
   */
  def normalize01(ranges: INDArray): INDArray => INDArray = {
    val toRanges = vstack(zeros(1, ranges.size(1)), ones(1, ranges.size(1)))
    transform(ranges, toRanges)
  }

  /**
   * Returns the normalizer function of row vector
   * The function returns ranges 0, 1 for defined ranges
   *
   * @param fromRanges the range of transformation the first row contains minimum values
   *                   and the second row contains the maximum values
   * @param toRanges   the range of transformation the first row contains minimum values
   *                   and the second row contains the maximum values
   */
  def transform(fromRanges: INDArray, toRanges: INDArray): INDArray => INDArray = {
    val m = toRanges.getRow(1).sub(toRanges.getRow(0)).divi(fromRanges.getRow(1).sub(fromRanges.getRow(0)))
    val q = m.mul(fromRanges.getRow(0)).subi(toRanges.getRow(0)).negi()
    x =>
      clip(x, fromRanges, copy = true).mul(m).addi(q)
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
   * Returns the distribution function for linear preferences
   *
   * @param x the preferences
   */
  def df(x: INDArray): INDArray = x.div(x.sumNumber().doubleValue())

  /**
   * Returns the egreedy policy for the given action values
   *
   * @param q       actions values
   * @param epsilon epsilon
   */
  def egreedy(q: INDArray, epsilon: INDArray): INDArray = {
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
