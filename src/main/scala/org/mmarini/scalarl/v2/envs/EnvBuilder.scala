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

package org.mmarini.scalarl.v2.envs

import com.typesafe.scalalogging.LazyLogging
import io.circe.ACursor
import org.mmarini.scalarl.v2.Env

/**
 *
 */
object EnvBuilder extends LazyLogging {
  /**
   *
   * @param conf the json configuration
   */
  def fromJson(conf: ACursor): Env = {
    val landerConf = LanderConf(conf)
    val coder = conf.get[String]("type").toOption match {
      case Some("Lander") =>
        LanderCustomCoder(conf)
      case Some("LanderTiles") =>
        LanderTilesEncoder(conf)
      case Some("LanderContinuous") =>
        LanderContinuousEncoder(conf)
      case Some(typ) => throw new IllegalArgumentException(s"Unreconginzed coder type '$typ'")
      case _ => throw new IllegalArgumentException("Missing coder type")
    }
    LanderStatus(landerConf, coder)
  }
}