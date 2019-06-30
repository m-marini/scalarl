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

import io.circe.Json

trait TraceMode {
  def traceBuilder(key: String): OperationBuilder
  def toJson: Json
}

object NoneTraceMode extends TraceMode {

  def traceBuilder(key: String): OperationBuilder = OperationBuilder()

  lazy val toJson = Json.obj(
    "mode" -> Json.fromString("NONE"))
}

case class AccumulateTraceMode(lambda: Double, gamma: Double) extends TraceMode {

  def traceBuilder(key: String): OperationBuilder = {
    val mul = lambda * gamma

    OperationBuilder(data =>
      data.get(s"${key}.trace").map(trace => {
        val feedback = data(s"${key}.feedback")
        val noClearTrace = data("noClearTrace")
        val newTrace = trace.mul(mul).muli(noClearTrace).addi(feedback)
        data +
          (s"${key}.trace" -> newTrace) +
          (s"${key}.feedback" -> newTrace)
      }).getOrElse(data))
  }

  lazy val toJson = Json.obj(
    "mode" -> Json.fromString("ACCUMULATE"),
    "lambda" -> Json.fromDoubleOrNull(lambda),
    "gamma" -> Json.fromDoubleOrNull(gamma))
}

object TraceMode {
  def fromJson(json: Json): TraceMode =
    json.hcursor.get[String]("mode") match {
      case Right("NONE")       => NoneTraceMode
      case Right("ACCUMULATE") => accumulateFromJson(json)
      case Right(x)            => throw new IllegalArgumentException(s"""trace mode "${x}" invalid""")
      case _                   => throw new IllegalArgumentException("trace mode not found")
    }

  private def accumulateFromJson(json: Json) = {
    val lambda = json.hcursor.get[Double]("lambda") match {
      case Right(x) => x
      case _        => throw new IllegalArgumentException("lambda not found")
    }
    val gamma = json.hcursor.get[Double]("gamma") match {
      case Right(x) => x
      case _        => throw new IllegalArgumentException("gamma not found")
    }
    AccumulateTraceMode(lambda = lambda, gamma = gamma)
  }
}
