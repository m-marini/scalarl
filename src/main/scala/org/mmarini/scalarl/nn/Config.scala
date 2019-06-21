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

import scala.PartialFunction.OrElse

case class Config(
  parent:     Option[Config],
  _eta:       Option[Double],
  _gamma:     Option[Double],
  _lambda:    Option[Double],
  _beta1:     Option[Double],
  _beta2:     Option[Double],
  _optimizer: Option[UpdaterFactory],
  _updater:   Option[UpdaterFactory]) {

  def eta: Option[Double] = _eta.orElse(parent.flatMap(_.eta))
  def gamma: Option[Double] = _gamma.orElse(parent.flatMap(_.gamma))
  def lambda: Option[Double] = _gamma.orElse(parent.flatMap(_.lambda))
  def beta1: Option[Double] = _beta1.orElse(parent.flatMap(_.beta1))
  def beta2: Option[Double] = _beta2.orElse(parent.flatMap(_.beta2))
  def optimzer: Option[UpdaterFactory] = _optimizer.orElse(parent.flatMap(_.optimzer))
  def updater: Option[UpdaterFactory] = _updater.orElse(parent.flatMap(_.updater))
}
