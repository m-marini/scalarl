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

import org.nd4j.linalg.api.ndarray.INDArray

/*
 * learning phase:
 * 1. calcolo degli outputs di tutti i layers (forward)
 * 2. calcolo di tutti i gradienti
 * 3. calcolo dell'errore nelle uscite (propagazione indietro degli errori)
 * 4. calcolo aggiorna i gradienti con Adams e limitatori
 * 5. calcolo nuovi valori di traces con i gradienti e errori
 * 6. calcolo aggiornamenti dei parametri rete e parametri di learning (algoritmi di aggiornamento, Adams, eligibility traces)
 */

/**
 * Computes the outputs for the inputs and change data parameter to fit the labels
 */
trait Network {
  /** Returns the data with computed outputs */
  def forward(data: NetworkData, inputs: INDArray): INDArray

  /** Returns the data with changed parameters to fit the labels */
  def fit(
           data: NetworkData,
           inputs: INDArray,
           labels: INDArray,
           mask: INDArray,
           noClearTrace: INDArray): NetworkData
}
