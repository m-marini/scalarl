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

import monix.eval.Task
import monix.reactive.Observable
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j

import java.io._

object FileUtils {

  /**
   * Write data to file
   *
   * @param file   the file
   * @param append true if append write
   * @param f      writer function
   */
  def withFile(file: File, append: Boolean)(f: Writer => Unit) {
    withWriter(new FileWriter(file, append))(f)
  }

  def withWriter(w: Writer)(f: Writer => Unit) {
    try {
      f(w)
    } finally {
      w.close()
    }
  }

  def writeINDArray(matrix: INDArray)(fw: Writer) {
    val Array(n, m) = matrix.shape()
    for {
      i <- 0L until n
    } {
      val record = for {j <- 0L until m} yield matrix.getDouble(i, j).toString
      fw.write(record.mkString(",") + "\n")
    }
  }

  def readINDArray(file: File): Observable[INDArray] =
    readFile(file).map(_.split(",").map(_.toDouble)).foldLeft(Array[Array[Double]]()) {
      case (s, v) => s :+ v
    }.map(Nd4j.create)

  def readFile(file: File): Observable[String] =
    Observable.fromLinesReader(Task.eval(new BufferedReader(new FileReader(file))))
}
