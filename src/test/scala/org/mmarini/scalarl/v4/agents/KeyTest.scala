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

package org.mmarini.scalarl.v4.agents

import com.typesafe.scalalogging.LazyLogging
import org.nd4j.linalg.factory.Nd4j._
import org.scalatest.{FunSpec, Matchers}

import scala.concurrent.duration.{DurationInt, FiniteDuration}

class KeyTest extends FunSpec with Matchers with LazyLogging {
  val Duration: FiniteDuration = 1 seconds
  val MinCount = 100

  create()

  describe("Test Seq[Int]") {
    val v29v0 = for (i <- 0 to 28) yield i
    val v30v0 = for (i <- 0 to 29) yield i
    val v30v1 = for (i <- 0 to 29) yield i
    val v1v0 = Seq(0)
    val v1v1 = Seq(0)
    val v1v2 = Seq(1)

    val k29v0 = ModelKey(v29v0)
    val k30v0 = ModelKey(v30v0)
    val k30v1 = ModelKey(v30v1)

    val k1v0 = ModelKey(v1v0)
    val k1v1 = ModelKey(v1v1)
    val k1v2 = ModelKey(v1v2)

    val timer = Profiler().run(Duration, MinCount) {
      v30v0 == v30v1
    }.timer

    val timer1 = Profiler().run(Duration, MinCount) {
      k30v0 == k30v1
    }.timer
    logger.info("equal v30 = {}, k30 = {}", timer.avg, timer1.avg)

    val timer2 = Profiler().run(Duration, MinCount) {
      v30v0 == v29v0
    }.timer

    val timer3 = Profiler().run(Duration, MinCount) {
      k30v0 == k29v0
    }.timer
    logger.info("not equal v30 = {}, k30 = {}", timer2.avg, timer3.avg)

    val timer4 = Profiler().run(Duration, MinCount) {
      v1v0 == v1v1
    }.timer

    val timer5 = Profiler().run(Duration, MinCount) {
      k1v0 == k1v1
    }.timer
    logger.info("equal v1 = {}, k1 = {}", timer4.avg, timer5.avg)

    val timer6 = Profiler().run(Duration, MinCount) {
      v1v0 == v1v2
    }.timer

    val timer7 = Profiler().run(Duration, MinCount) {
      k1v0 == k1v2
    }.timer
    logger.info("not equal v1 = {}, k1 = {}", timer6.avg, timer7.avg)

    it("should improve the inequality") {
      timer2.avg should be > timer3.avg
      timer6.avg should be > timer7.avg
    }
  }
}
