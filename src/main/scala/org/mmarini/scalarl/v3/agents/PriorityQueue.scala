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

package org.mmarini.scalarl.v3.agents

import io.circe.ACursor
import org.nd4j.linalg.api.ndarray.INDArray

/**
 * Prioriry queue with insertion threshold
 *
 * @param threshold the threshold
 * @param queue     the data queue
 * @tparam T the type of elements
 */
case class PriorityQueue[T](threshold: Double, queue: Map[T, Double]) {

  /**
   * Returns the priority queue.
   * Keeps only the given keys
   *
   * @param keys the kept keys
   */
  def keep(keys: Set[T]): PriorityQueue[T] = {
    val newQueue = queue.filter(entry => keys.contains(entry._1))
    copy(queue = newQueue)
  }

  /** Returns the value and the new queue without the value */
  def dequeue(): (Option[T], PriorityQueue[T]) = if (queue.isEmpty) {
    (None, this)
  } else {
    val value = queue.toSeq.sortBy(_._2).lastOption.map(_._1)
    val newQueue = value.map(k => copy(queue = queue - k)).getOrElse(this)
    (value, newQueue)
  }

  /**
   * Returns the queue with new entry if score higher then threshold
   *
   * @param entry the entry
   */
  def +(entry: (T, Double)): PriorityQueue[T] = entry match {
    case (_, score) if score <= threshold =>
      this
    case (value, score) =>
      copy(queue = queue + (value -> score))
  }
}

/** The object factory of [[PriorityQueue]] */
object PriorityQueue {
  /**
   * Returns the priority queue from json configuration
   *
   * @param conf the configuration
   */
  def fromJson(conf: ACursor): PriorityQueue[(INDArray, INDArray)] = PriorityQueue(
    threshold = conf.get[Double]("scoreThreshold").toTry.get,
    queue = Map())
}
