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

import monix.reactive.Observable
import org.mmarini.scalarl.v6.agents.AgentEvent
import org.mmarini.scalarl.v6.{Session, Step}
import org.nd4j.linalg.api.ndarray.INDArray

object Implicits {
  /**
   * Returns the steps wrapper from a agent events observable
   *
   * @param agentEvent the observable
   */
  implicit def from(agentEvent: Observable[AgentEvent]): AgentEventWrapper = new AgentEventWrapper(agentEvent)

  /**
   * Returns the steps wrapper from a session
   *
   * @param session the session
   */
  implicit def from(session: Session): StepsWrapper = new StepsWrapper(session.steps)

  /**
   *
   * @param obs the observable
   */
  implicit def from(obs: Observable[Step]): StepsWrapper = new StepsWrapper(obs)

  /**
   *
   * @param obs the observable
   */
  implicit def from(obs: Observable[INDArray]): INDArrayWrapper = new INDArrayWrapper(obs)

  /**
   *
   * @param obs the observable
   */
  implicit def from(obs: Observable[(Step, INDArray)]): MonitorWrapper = new MonitorWrapper(obs)
}
