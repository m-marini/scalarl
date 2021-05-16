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

package org.mmarini.scalarl.v6.agents

import io.circe.ACursor
import org.nd4j.linalg.learning.config.{Adam, IUpdater, Sgd}

/**
 * Builder of [[IUpdater]] from json configuration
 */
object UpdaterBuilder {

  /**
   * Returns an [[IUpdater]] builder from parameters counter
   *
   * @param conf     the json configuration
   * @param noParams the number of parameters
   */
  def fromJson(conf: ACursor)(noParams: => Int): IUpdater = conf.get[String]("updater").toTry.get match {
    case "Adam" => adam(conf)(noParams)
    case "Sgd" => sgd(conf)(noParams)
    case x => throw new IllegalArgumentException(s"unrecognized updater $x")
  }

  /**
   * Returns an [[Adam]] builder from parameters counter
   *
   * @param conf     the json configuration
   * @param noParams the number of parameters
   */
  def adam(conf: ACursor)(noParams: => Int): Adam = {
    val eta = learningRate(conf)(noParams)
    val beta1 = conf.get[Double]("beta1").toTry.get
    val beta2 = conf.get[Double]("beta2").toTry.get
    val epsilon = conf.get[Double]("epsilonAdam").toTry.get
    new Adam(eta, beta1, beta2, epsilon)
  }

  /**
   * Returns an [[Sgd]] builder from parameters counter
   *
   * @param conf     the json configuration
   * @param noParams the number of parameters
   */
  def sgd(conf: ACursor)(noParams: => Int): Sgd = new Sgd(learningRate(conf)(noParams))

  /**
   * Returns the learningRate builder from parameters counts
   *
   * @param conf     the json configuration
   * @param noParams the number of parameters
   */
  def learningRate(conf: ACursor)(noParams: => Int): Double =
    conf.get[Double]("learningRate").toOption.getOrElse {
      val eta = conf.get[Double]("autoScaleLearningRate").toTry.get
      eta / noParams
    }
}
