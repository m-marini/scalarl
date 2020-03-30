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

import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.ops.transforms.Transforms

/**
 * Computes the outputs for the inputs and change data parameter to fit the labels
 */
trait OperationBuilder {
  def build: Operation

  def then(f1: Operation): OperationBuilder

  def then(u: OperationBuilder): OperationBuilder
}

object OperationBuilder {
  def apply(f: Operation): OperationBuilder = new MonoOperationBuilder(f)

  def apply(): OperationBuilder = IdentityBuilder

  def thetaBuilder(key: String, constrainAllParms: Option[Double]): OperationBuilder = {
    val feedbackKey = s"${key}.feedback"
    val thetaKey = s"${key}.theta"
    val thetaDeltaKey = s"${key}.thetaDelta"
    val constrain = constrainAllParms.map(max =>
      (data: INDArray) =>
        Transforms.hardTanh(data.div(max)).mul(max)).
      getOrElse((data: INDArray) => data)

    new MonoOperationBuilder(data =>
      data.get(feedbackKey).map(feedback => {
        val theta = data(thetaKey)
        val thetaDelta = data(thetaDeltaKey)
        val newTheta = constrain(theta.add(feedback.mul(thetaDelta)))
        val constraint = constrain(newTheta)

        Sentinel(constraint, thetaKey)

        data + (thetaKey -> constraint)
      }).getOrElse(data))
  }
}

class MonoOperationBuilder(f: Operation) extends OperationBuilder {
  def build = f

  def then(u: OperationBuilder): OperationBuilder =
    if (u == IdentityBuilder) this else then(u.build)

  def then(f1: Operation): OperationBuilder =
    new MonoOperationBuilder(data => f1(f(data)))
}

object IdentityBuilder extends OperationBuilder {
  val build = data => data

  def then(u: OperationBuilder): OperationBuilder =
    if (u == IdentityBuilder) this else then(u.build)

  def then(f1: NetworkData => NetworkData): OperationBuilder = new MonoOperationBuilder(f1)
}
