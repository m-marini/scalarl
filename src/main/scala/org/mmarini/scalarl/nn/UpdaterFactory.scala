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

package org.mmarini.scalarl.nn

/**
 *
 */
object UpdaterFactory {
  /**
   *
   */
  val identityUpdater = (data: LayerData) => data

  val paramsUpdater = (layer: Layer) => layer match {
    case _: DenseLayer => (data: LayerData) => {
      //      val delta = localData.delta
      //      val dParms = localData.dParms
      //      val parms = localData.parms.add(dParms)
      //      localData.copy(parms)
      ???
    }
    case _ => identityUpdater
  }

  //  def adamOptimizer(alpha: Double, beta1: Double, beta2: Double) = (layer: Layer) => identityUpdater

  def sgdUpdater(alpha: Double): UpdaterFactory = (layer: Layer) => layer match {
    case _: DenseLayer => (data: LayerData) => {
      //      val localData = data.asInstanceOf[DenseLayerData]
      //      val grad = localData.gradient
      //      val dParms = grad.mul(alpha)
      //      localData.copy(dParms = dParms)
      ???
    }
    case _ => identityUpdater
  }

  def traceUpdater(gamma: Double, lambda: Double): UpdaterFactory = (layer: Layer) => layer match {
    case denseLayer: DenseLayer => (data: LayerData) => {
      //      val localData = data.asInstanceOf[DenseLayerData]
      //      val traces = localData.traces
      //      val dParms = localData.dParms
      //
      //      // E' = E lambda gamma + dParms
      //      val newTrace = traces.mul(lambda * gamma).addi(dParms)
      //      localData.copy(traces = newTrace, dParms = newTrace)
      ???
    }
    case _ => identityUpdater
  }

  def compose(factories: Seq[UpdaterFactory]): UpdaterFactory = (layer: Layer) =>
    factories.map(factory => factory(layer)).foldLeft(identityUpdater)((acc, updater) =>
      (data: LayerData) =>
        updater(acc(data)))
}
