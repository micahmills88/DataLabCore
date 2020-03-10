using ILGPU;
using ILGPU.Runtime;
using System;
using System.Collections.Generic;
using System.Text;

namespace DataLabCore.Kernels
{
    public class DenseKernels
    {
        #region forward kernels
        private void DenseForward(
            ILGPU.Index index,
            ArrayView<float> input,
            ArrayView<float> weights,
            ArrayView<float> bias,
            ArrayView<float> output,
            int input_columns,
            int out_columns
        )
        {
            int outRow = index / out_columns;
            int outColumn = index % out_columns;
            float sum = 0f;
            for (int c = 0; c < input_columns; c++)
            {
                int leftIndex = outRow * input_columns + c;
                int rightIndex = outColumn + (c * out_columns);
                sum += input[leftIndex] * weights[rightIndex];
            }
            output[index] = sum + bias[index];
        }
        #endregion forward kernels

        #region backward kernels
        private void CalculateErrors(ILGPU.Index index, ArrayView<float> results, ArrayView<float> errors, ArrayView<float> outputs, int outRows, int outCols)
        {
            //inputs will be derived already
            //transpose the inputs as you multiply them

        }
        #endregion backward kernels
    }
}
