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

package org.mmarini.scalarl.v4.agents

import io.circe.ACursor
import org.mmarini.scalarl.v4.Feedback
import org.nd4j.linalg.api.ndarray.INDArray

/**
 * The model structure
 *
 * @param minModelSize the lower size of model
 * @param maxModelSize the higher size of model
 * @param data         the data of model
 * @param ordering     the ordering to remove less referenced values
 * @tparam K the key
 * @tparam V the value
 */
case class Model[K, V](minModelSize: Int,
                       maxModelSize: Int,
                       data: Map[K, V],
                       ordering: Ordering[(K, V)]) {

  /**
   * Returns the value for a key
   *
   * @param key the key
   */
  def get(key: K): Option[V] = data.get(key)

  /**
   * Returns the set of keys for a validated values
   *
   * @param predicate the matching function
   */
  def filterValues(predicate: V => Boolean): Map[K, V] = {
    val result = data.filter(entry => predicate(entry._2))
    result
  }

  /**
   * Returns the model with a value for a key
   *
   * @param entry the entry
   */
  def +(entry: (K, V)): Model[K, V] = {
    val data1 = if (data.size >= maxModelSize) {
      // shrink data removing the lower ones
      val sorted = data.toSeq.sorted(ordering)
      val shrunk = sorted.take(minModelSize)
      shrunk.toMap
    } else {
      data
    }
    copy(data = data1 + entry)
  }
}

/** The object factory of [[Model]] */
object Model {
  /**
   * Returns the model from json configuration
   *
   * @param conf the configuration
   */
  def fromJson(conf: ACursor): Model[(INDArray, INDArray), Feedback] = {
    val ordering = Ordering.by((t: ((INDArray, INDArray), Feedback)) => t._2.s0.time.getDouble(0L)).reverse
    Model(
      minModelSize = conf.get[Int]("minModelSize").toTry.get,
      maxModelSize = conf.get[Int]("maxModelSize").toTry.get,
      data = Map(),
      ordering = ordering
    )
  }
}
