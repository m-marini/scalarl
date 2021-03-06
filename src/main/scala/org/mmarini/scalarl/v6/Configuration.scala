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

import io.circe.yaml.parser
import io.circe.{ACursor, Json}
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j.{create, ones}

import java.io.{FileReader, Reader, StringReader}
import scala.util.Try

object Configuration {

  /**
   * Returns the json from a string
   *
   * @param text the string
   */
  def jsonFormString(text: String): Json = jsonFromReader(new StringReader(text))

  /**
   * Returns the json from a file
   *
   * @param file the file
   */
  def jsonFromFile(file: String): Json = jsonFromReader(new FileReader(file))

  /**
   * Returns the json from a Reader
   *
   * @param reader the reader
   */
  def jsonFromReader(reader: Reader): Json = parser.parse(reader).toTry.get

  /**
   * Returns a scalar from json
   *
   * @param conf the json
   */
  def scalarFromJson(conf: ACursor): Try[INDArray] =
    conf.as[Double].toTry.map(ones(1).mul(_))

  /**
   * Returns a vector from json
   *
   * @param conf the json
   * @param n    the number of dimensions
   */
  def vectorFromJson(conf: ACursor)(n: Int): Try[INDArray] =
    conf.as[Array[Double]].toTry.flatMap(data => Try {
      require(data.length == n, s"vector has ${
        data.length
      } elements: must have $n elements")
      create(data)
    })

  /**
   * Returns a positive integer vector from json
   *
   * @param conf the json
   * @param n    the number of dimensions
   */
  def posIntVectorFromJson(conf: ACursor)(n: Int): Try[INDArray] =
    conf.as[Array[Int]].toTry.flatMap(data => Try {
      require(data.length == n, s"vector has ${
        data.length
      } elements: must have $n elements")
      for {i <- 0 until n} {
        require(data(i) > 0, s"dimensions $i has value ${data(i)}: it must be positive")
      }
      create(data.map(_.toDouble))
    })

  /**
   * Returns ranges from Json configuration
   *
   * @param conf       the json configuration
   * @param dimensions the number of dimensions
   */
  def rangesFromJson(conf: ACursor)(dimensions: Int): Try[INDArray] =
    for {
      data <- matrixFromJson(conf)(dimensions, 2)
      _ <- Try {
        for {
          i <- 0 until dimensions
        } {
          require(data.getDouble(i, 0L) <= data.getDouble(i, 1L), s"range($i) is ${
            data.getRow(i)
          }: must be min, max")
        }
      }
    } yield data.transpose()

  /**
   * Returns a matrix from Json configuration
   *
   * @param conf the json configuration
   * @param n    number of rows
   * @param m    number of columns
   */
  def matrixFromJson(conf: ACursor)(n: Int, m: Int): Try[INDArray] =
    conf.as[Array[Array[Double]]].toTry.flatMap(data => Try {
      require(data.length == n, s"vector has ${data.length} elements: must have $n elements")
      for {
        (range, i) <- data.zipWithIndex
      } {
        require(range.length == 2, s"ranges($i) has ${range.length} elements: must have 2 elements")
      }
      create(data)
    })
}
