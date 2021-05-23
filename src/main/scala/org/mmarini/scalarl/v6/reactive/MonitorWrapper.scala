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

import monix.eval.Task
import monix.reactive.Observable
import org.mmarini.scalarl.v6.Step
import org.mmarini.scalarl.v6.agents.{ActorCriticAgent, PriorityPlanner}
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j._

/**
 * Wrapper of [[Observable[INDArray]]] to add functionalities
 *
 * @param observable the observable
 */
class MonitorWrapper(val observable: Observable[(Step, INDArray)]) extends ObservableWrapper[(Step, INDArray)] {
  /** Returns the observable that log as info the monitor data */
  def logInfo(): MonitorWrapper = new MonitorWrapper(observable.doOnNext(data => Task.eval {
    val (step, avg) = data
    val modelSize = step.agent0 match {
      case a: ActorCriticAgent => a.planner match {
        case Some(planner: PriorityPlanner[INDArray, INDArray]) => planner.model.size
        case _ => 0
      }
      case _ => 0
    }
    val rew = step.agent0 match {
      case actor: ActorCriticAgent => actor.avg
      case _ => avg.getColumn(0)
    }
    val alphas = step.agent0 match {
      case actor: ActorCriticAgent => hstack(actor.alpha: _*)
      case _ => zeros(1)
    }
    logger.info(f"Epoch ${
      step.epoch
    }%,3d, Steps ${
      step.step
    }%5d ,avg rewards=${
      rew.getDouble(0L)
    }%12g, avgScore=${
      avg.getDouble(1L)
    }%12g, model=$modelSize%4d, alpha=${
      alphas
    }")
  }))
}
