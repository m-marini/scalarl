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

trait LossFunction {
  def toJson: Json

  def lossBuilder: OperationBuilder

  def deltaBuilder: OperationBuilder
}

object MSELossFunction extends LossFunction {
  lazy val toJson = Json.fromString("MSE")

  lazy val deltaBuilder = OperationBuilder(data => {
    val outputs = data("outputs")
    val labels = data("labels")
    val mask = data("mask")
    val delta = labels.sub(outputs).muli(mask)
    data + ("delta" -> delta)
  })

  lazy val lossBuilder = OperationBuilder(data => {
    val outputs = data("outputs")
    val labels = data("labels")
    val mask = data("mask")
    val diff = outputs.sub(labels).muli(mask)
    val loss = diff.muli(diff).sum(1)
    data + ("loss" -> loss)
  })
}

object LossFunction {
  def fromJson(json: Json): LossFunction = json.asString match {
    case Some("MSE") => MSELossFunction
    case Some(x)     => throw new IllegalArgumentException(s"""loss function "${x}" illegal""")
    case _           => throw new IllegalArgumentException("missing loss function")
  }
}