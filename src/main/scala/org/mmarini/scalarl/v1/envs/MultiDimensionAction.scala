package org.mmarini.scalarl.v1.envs

import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j

class MultiDimensionAction(sizes: Int*) {
  require(!sizes.isEmpty)
  sizes.foreach(x => require(x > 0))

  val actions = sizes.reduce(_ * _)

  val strides = sizes.init.foldLeft(Seq(1)) {
    case (s, l) => s :+ (s.last * l)
  }

  /**
   * Returns the multi dimension action vector
   *
   * @param action the action
   */
  def vector(action: Int): INDArray = {
    require(action >= 0 && action < actions)
    val v = for {(s, l) <- strides.zip(sizes)} yield
      ((action / s) % l).toDouble
    Nd4j.create(v.toArray)
  }
}
