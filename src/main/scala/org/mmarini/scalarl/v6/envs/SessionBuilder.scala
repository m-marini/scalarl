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

package org.mmarini.scalarl.v6.envs

import com.typesafe.scalalogging.LazyLogging
import io.circe.ACursor
import monix.execution.Scheduler
import monix.execution.Scheduler.global
import monix.reactive.Observable
import org.mmarini.scalarl.v6._
import org.mmarini.scalarl.v6.agents.AgentEvent
import org.mmarini.scalarl.v6.reactive.Implicits._

import java.io.File
import scala.util.Try

/**
 *
 */
object SessionBuilder extends LazyLogging {
  private implicit val scheduler: Scheduler = global
  private val SaveStepInterval = 10

  /**
   * Returns the session
   *
   * @param conf        the json configuration
   * @param epoch       the number of epoch
   * @param env         the environment
   * @param agent       the agent
   * @param agentEvents the agent event observable
   */
  def fromJson(conf: ACursor)(
    epoch: Int,
    kpiFileParam: Option[String],
    dumpFileParam: Option[String],
    traceFileParam: Option[String],
    env: => Env,
    agent: => Agent,
    agentEvents: Observable[AgentEvent]): Try[Session] = {

    for {
      numSteps <- conf.get[Int]("numSteps").toTry
    } yield {
      val dump = dumpFileParam.orElse(conf.get[String]("dump").toOption)
      val trace = traceFileParam.orElse(conf.get[String]("trace").toOption)
      val saveModel = conf.get[String]("modelFile").toOption
      val kpiFile = kpiFileParam.orElse(conf.get[String]("kpiFile").toOption)
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
        logger.info("Dump file {}", filename)
        session.landerDump().writeCsv(new File(filename)).subscribe()
      })

      trace.foreach(filename => {
        logger.info("Trace file {}", filename)
        session.landerTrace().writeCsv(new File(filename)).subscribe()
      })

      kpiFile.foreach(filename => {
        logger.info("Kpi file {}", filename)
        val kpisOnPlanning = conf.get[Boolean]("kpisOnPlanning")
        val agentEventWrapper = if (kpisOnPlanning.getOrElse(false)) {
          from(agentEvents)
        } else {
          agentEvents.sampleBySessionStep(session)
        }
        agentEventWrapper.kpis().writeCsv(new File(filename)).subscribe()
      })

      // Save model every 10 steps
      saveModel.foreach(file => {
        logger.info("Model file {}", file)
        val path = new File(file)
        session.steps.takeEveryNth(SaveStepInterval).saveAgent(path).subscribe()
        session.steps.last.saveAgent(path).subscribe()
      })
      session
    }
  }
}