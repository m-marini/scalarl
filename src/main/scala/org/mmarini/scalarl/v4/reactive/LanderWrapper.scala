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

package org.mmarini.scalarl.v4.reactive

import monix.eval.Task
import monix.reactive.Observable
import org.mmarini.scalarl.v4.envs.LanderStatus

/**
 * Wrapper of [[Observable[INDArray]]] to add functionalities
 *
 * @param observable the observable
 */
class LanderWrapper(val observable: Observable[LanderStatus]) extends ObservableWrapper[LanderStatus] {
  /** Returns the final state observable */
  def filterFinal(): LanderWrapper = new LanderWrapper(observable.filter(_.isFinal))

  /** Return the status logged observable */
  def logInfo(): LanderWrapper = new LanderWrapper(observable.doOnNext(lander => Task.eval {
    logger.whenInfoEnabled {
      val dis = lander.distance.getDouble(0l)
      val dir = Math.round(Math.toDegrees(lander.direction.getDouble(0l) + 2 * Math.PI)) % 360
      val hs = lander.hSpeed.getDouble(0l)
      val vs = lander.vSpeed.getDouble(0l)
      val alt = lander.height.getDouble((0l))
      val sd = Math.round(Math.toDegrees(lander.speedDirection.getDouble(0l) + 2 * Math.PI)) % 360
      logger.info(f"${lander.status} D$dis%.0f R$dir alt=$alt%.0f hs=$hs%.1f R$sd vs=$vs%.1f")
    }
  }))
}
