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

package org.mmarini.scalarl.v3.envs

import java.io.File

import com.typesafe.scalalogging.LazyLogging
import io.circe.ACursor
import monix.execution.Scheduler
import monix.execution.Scheduler.global
import org.mmarini.scalarl.v3._
import org.mmarini.scalarl.v3.reactive.WrapperBuilder._


/**
 *
 */
object SessionBuilder extends LazyLogging {
  private implicit val scheduler: Scheduler = global
  private val SaveStepInterval = 10

  /**
   * Returns the session
   *
   * @param conf  the json configuration
   * @param epoch the number of epoch
   * @param env   the environment
   * @param agent the agent
   */
  def fromJson(conf: ACursor)(epoch: Int, env: => Env, agent: => Agent): Session = {
    val numSteps = conf.get[Int]("numSteps").toTry.get
    val dump = conf.get[String]("dump").toOption
    val trace = conf.get[String]("trace").toOption
    val saveModel = conf.get[String]("modelFile").toOption
    val kpiFile = conf.get[String]("kpiFile").toOption

    // Clean up all files
    if (epoch == 0) {
      (dump.toSeq ++ trace ++ saveModel ++ kpiFile).foreach(new File(_).delete())
    }
    saveModel.foreach(f => {
      new File(f).mkdirs()
    })

    // Create session
    val session = new Session(
      numSteps = numSteps,
      env = env,
      agent = agent,
      epoch = epoch)

    dump.foreach(filename => {
      session.landerDump().writeCsv(new File(filename)).subscribe()
    })

    trace.foreach(filename => {
      session.landerTrace().writeCsv(new File(filename)).subscribe()
    })

    kpiFile.foreach(filename => {
      session.kpis().writeCsv(new File(filename)).subscribe()
    })

    // Save model every 10 steps
    saveModel.foreach(file => {
      val path = new File(file)
      session.steps.takeEveryNth(SaveStepInterval).saveAgent(path).subscribe()
      session.steps.last.saveAgent(path).subscribe()
    })
    session
  }
}