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

package org.mmarini.scalarl.v6.reactive

import monix.eval.Task
import monix.reactive.Observable
import org.mmarini.scalarl.v6.FileUtils.{withFile, writeINDArray}
import org.nd4j.linalg.api.ndarray.INDArray

import java.io.File

/**
 * Wrapper of [[Observable[INDArray]]] to add functionalities
 *
 * @param observable the observable
 */
class INDArrayWrapper(val observable: Observable[INDArray]) extends ObservableWrapper[INDArray] {
  /**
   * Returns the observable that writes in a csv file
   *
   * @param file the file to write
   */
  def writeCsv(file: File): INDArrayWrapper = new INDArrayWrapper(observable.doOnNext(rows => Task.eval {
    withFile(file, append = true)(writeINDArray(rows))
  }))

  /** Returns the observable that log as info the row */
  def logDebug(): INDArrayWrapper = new INDArrayWrapper(observable.doOnNext(rows => Task.eval {
    logger.debug("{}", rows)
  }))

  /** Returns the observable that log as info the row */
  def logInfo(): INDArrayWrapper = new INDArrayWrapper(observable.doOnNext(rows => Task.eval {
    logger.info("{}", rows)
  }))
}
