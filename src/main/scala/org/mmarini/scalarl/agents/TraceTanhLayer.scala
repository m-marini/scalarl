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

package org.mmarini.scalarl.agents

import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.indexing.NDArrayIndex
import org.nd4j.linalg.indexing.INDArrayIndex
import org.nd4j.linalg.api.ops.impl.transforms.Tanh

/**
 */
class TraceTanhLayer extends TraceLayer {

  /** Returns the output of layer given an input */
  override def forward(input: INDArray): INDArray = {
    val y = Nd4j.getExecutioner().execAndReturn(new Tanh(input.dup()))
    y
  }

  /**
   * Returns the backward errors and mask after updating the layer parameters given the input, output, output, errors
   * and output mask
   */
  override def backward(input: INDArray, output: INDArray, errors: INDArray, mask: INDArray): (INDArray, INDArray) = {
    val inpError = output.sub(1.0).muli(output.add(1.0)).negi().muli(errors).muli(mask)
    (inpError, mask)
  }

  override def clearTraces(): TraceLayer = this
}
