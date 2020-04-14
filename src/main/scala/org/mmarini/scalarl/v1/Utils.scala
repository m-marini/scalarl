package org.mmarini.scalarl.v1

import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.api.rng.Random
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.indexing.NDArrayIndex
import org.nd4j.linalg.ops.transforms.Transforms

object Utils {

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
    result
  }

  /**
   * Returns random integer for a distribution function
   *
   * @param x the distribution function
   */
  def randomInt(x: INDArray): Random => Int = cdfRandomInt(cdf(x))

  /**
   * Returns the distribution function for linear preferences
   *
   * @param x the preferences
   */
  def df(x: INDArray): Policy = x.div(x.sumNumber().doubleValue())

  /**
   * Returns the softMax distribution
   *
   * @param x the preferences
   */
  def softMax(x: INDArray): Policy = df(Transforms.exp(x))

  /**
   * Returns
   * /**
   * * Returns the probabilities for the actions in egreedy policy
   * * The probability for greedy action is 1 - epsilon
   * * the probabilities for other actions are epsilon / (n - 1)
   * *
   * * @param q       the action values
   * * @param epsilon the epsilon parameter
   **/ random integer for a softMax distribution function
   *
   * @param x the softMax distribution function
   */
  def softMaxRandomInt(x: INDArray): Random => Int = randomInt(softMax(x))
  def egreedy(q: INDArray, epsilon: Double): Policy = {
    val n = q.length()
    if (n > 1) {
      val act = q.argMax().getInt(0)
      val result = Nd4j.ones(n).muli(epsilon / (n - 1))
      result.putScalar(act, 1 - epsilon)
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
    x.get(idx)
  }

  /**
   * Returns the status value for the action and argMax
   * The status value of a policy is the max value for the channel
   *
   * @param q      action value
   * @param action action
   * @return the array of values for each channel and the indices of max action features
   */
  def v(q: Policy, action: Action): VValue = q.getDouble(action.toLong)

  /**
   * Returns the expected status value
   *
   * @param q  the action values
   * @param pi the policy
   */
  def vExp(q: QValues, pi: Policy): Double = q.mul(pi).sumNumber().doubleValue()

  /**
   * Returns the greedy status value
   *
   * @param q       the actions values
   * @param actions the valid actions
   */
  def vStar(q: QValues): Double = q.maxNumber().doubleValue()
}
