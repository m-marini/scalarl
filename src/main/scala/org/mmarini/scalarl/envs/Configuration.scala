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

package org.mmarini.scalarl.envs

import java.io.FileReader

import scala.collection.JavaConversions.`deprecated asScalaBuffer`
import scala.collection.JavaConversions.`deprecated mapAsScalaMap`

import org.yaml.snakeyaml.Yaml
import java.io.Reader

class Configuration(conf: Map[String, Any]) {
  def getConf(key: String): Configuration = conf.get(key) match {
    case Some(m: java.util.Map[_, _]) => new Configuration(m.toMap.asInstanceOf[Map[String, Any]])
    case _                            => new Configuration(Map())
  }

  def getNumber(key: String): Option[Number] = conf.get(key) match {
    case Some(n: Number) => Some(n)
    case _               => None
  }

  def getInt(key: String): Option[Int] = getNumber(key).map(_.intValue())

  def getLong(key: String): Option[Long] = getNumber(key).map(_.longValue())

  def getDouble(key: String): Option[Double] = getNumber(key).map(_.doubleValue())

  def getString(key: String): Option[String] = conf.get(key) match {
    case Some(s: String) => Some(s)
    case _               => None
  }

  def getList[T](key: String): List[T] = {
    val x = conf.get(key)
    x match {
      case Some(l: java.util.List[_]) => l.asInstanceOf[java.util.List[T]].toList
      case _                          => List()
    }
  }

  def getConfList(key: String): List[Configuration] = {
    val x = conf.get(key)
    x match {
      case Some(l: java.util.List[_]) => l.asInstanceOf[java.util.List[java.util.Map[String, Any]]].toList.map(c => new Configuration(c.toMap))
      case _                          => List()
    }
  }
}

object Configuration {

  def fromFile(file: String): Configuration = fromReader(new FileReader(file))

  def fromReader(reader: Reader): Configuration = {
    val conf = new Yaml().load(reader)
    new Configuration(conf.asInstanceOf[java.util.Map[String, Any]].toMap)
  }
}
