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

package org.mmarini.scalarl

import java.io._

import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import rx.lang.scala.Observable

object FileUtils {

  def withFile(file: String, append: Boolean)(f: Writer => Unit) {
    withWriter(new FileWriter(file, true))(f)
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
    readFile(file).map(_.split(",").map(_.toDouble)).toArray.map(
      Nd4j.create)

  def readFile(file: File): Observable[String] =
    Observable(s => {
      try {
        val br = new BufferedReader(new FileReader(file))
        try {
          var reading = true

          do {
            val line = Option(br.readLine())
            if (line.isEmpty) {
              s.onCompleted()
              reading = false
            } else {
              s.onNext(line.get)
            }
          } while (reading)
        } finally {
          br.close()
        }
      } catch {
        case ex: Throwable => s.onError(ex)
      }
    })
}
