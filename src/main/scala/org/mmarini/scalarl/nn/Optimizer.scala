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

import org.nd4j.linalg.ops.transforms.Transforms

import io.circe.Json

trait Optimizer {

  def optimizeBuilder(key: String): OperationBuilder

  def toJson: Json
}

case class SGDOptimizer(alpha: Double) extends Optimizer {

  def optimizeBuilder(key: String): OperationBuilder = OperationBuilder(data => {
    val gradient = data.get(s"${key}.gradient")
    val feedback = gradient.map(g => g.mul(alpha))
    feedback.
      map(f => data + (s"${key}.feedback" -> f)).
      getOrElse(data)
  })

  lazy val toJson = Json.obj(
    "mode" -> Json.fromString("SGD"),
    "alpha" -> Json.fromDoubleOrNull(alpha))
}

case class AdamOptimizer(alpha: Double, beta1: Double, beta2: Double, epsilon: Double) extends Optimizer {
  val omb1 = 1 - beta1
  val omb2 = 1 - beta2
  def optimizeBuilder(key: String): OperationBuilder = OperationBuilder(data =>
    data.get(s"${key}.gradient").
      map(g => {
        val m1 = data(s"${key}.m1")
        val m2 = data(s"${key}.m2")
        val g2 = g.mul(g)
        // m1 = m1 b1 + g(1-b1)
        // m1^ = m1/(1-b1)
        val newM1 = m1.mul(beta1).addi(g.mul(omb1))
        val newM2 = m2.mul(beta2).addi(g2.mul(omb2))

        val m1Norm = newM1.div(omb1)
        val m2Norm = newM2.div(omb2)

        val feedback = m1Norm.divi(Transforms.sqrt(m2Norm).addi(epsilon)).muli(alpha)

        data +
          (s"${key}.m1" -> newM1) +
          (s"${key}.m2" -> newM2) +
          (s"${key}.feedback" -> feedback)
      }).getOrElse(data))

  lazy val toJson = Json.obj(
    "mode" -> Json.fromString("ADAM"),
    "alpha" -> Json.fromDoubleOrNull(alpha),
    "beta1" -> Json.fromDoubleOrNull(beta1),
    "beta2" -> Json.fromDoubleOrNull(beta2),
    "epsilon" -> Json.fromDoubleOrNull(epsilon))
}

object Optimizer {
  def fromJson(json: Json): Optimizer = json.asObject.flatMap(_("mode")).flatMap(_.asString) match {
    case Some("SGD")  => sgdFromJson(json)
    case Some("ADAM") => adamFromJson(json)
    case Some(x)      => throw new IllegalArgumentException(s"""optimizer mode "${x}" invalid""")
    case _            => throw new IllegalArgumentException("missing optimizer mode")
  }

  private def sgdFromJson(json: Json) =
    json.asObject.flatMap(_("alpha")).flatMap(_.asNumber).map(_.toDouble) match {
      case Some(x) => SGDOptimizer(x)
      case _       => throw new IllegalArgumentException("missing alpha")
    }

  private def adamFromJson(json: Json) = {
    val alpha = json.asObject.flatMap(_("alpha")).flatMap(_.asNumber).map(_.toDouble) match {
      case Some(x) => x
      case _       => throw new IllegalArgumentException("missing alpha")
    }
    val beta1 = json.asObject.flatMap(_("beta1")).flatMap(_.asNumber).map(_.toDouble) match {
      case Some(x) => x
      case _       => throw new IllegalArgumentException("missing beta1")
    }
    val beta2 = json.asObject.flatMap(_("beta2")).flatMap(_.asNumber).map(_.toDouble) match {
      case Some(x) => x
      case _       => throw new IllegalArgumentException("missing beta2")
    }
    val epsilon = json.asObject.flatMap(_("epsilon")).flatMap(_.asNumber).map(_.toDouble) match {
      case Some(x) => x
      case _       => throw new IllegalArgumentException("missing epsilon")
    }
    AdamOptimizer(alpha = alpha, beta1 = beta1, beta2 = beta2, epsilon = epsilon)
  }
}
