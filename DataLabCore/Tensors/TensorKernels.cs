using ILGPU;
using ILGPU.Algorithms;
using ILGPU.Runtime;
using System;
using System.Collections.Generic;
using System.Text;

namespace DataLabCore
{
    public class TensorKernels
    {
        public Action<ILGPU.Index, ArrayView<float>, ArrayView<float>, ArrayView<float>, int, int> MatrixMultiply;
        public Action<ILGPU.Index, ArrayView<float>, ArrayView<float>> AddBias;
        public Action<ILGPU.Index, ArrayView<float>> ActivateSigmoid;
        public Action<ILGPU.Index, ArrayView<float>> DeriveSigmoid;
        public Action<ILGPU.Index, ArrayView<float>, ArrayView<float>, int, int> Transpose2D;
        public Action<ILGPU.Index, ArrayView<float>, ArrayView<float>> MultiplyErrors;
        public Action<ILGPU.Index, ArrayView<float>, ArrayView<float>, ArrayView<float>, int, int> TransposedMatrixMultiply;
        public Action<ILGPU.Index, ArrayView<float>, ArrayView<float>, int> RowSums;
        public Action<ILGPU.Index, ArrayView<float>, ArrayView<float>, float> AdjustMomentum;
        public Action<ILGPU.Index, ArrayView<float>, ArrayView<float>, float> ApplyGradient;

        public TensorKernels(Accelerator accelerator)
        {
            MatrixMultiply = accelerator.LoadAutoGroupedStreamKernel<ILGPU.Index, ArrayView<float>, ArrayView<float>, ArrayView<float>, int, int>(K_Matrix_Multiply);
            AddBias = accelerator.LoadAutoGroupedStreamKernel<ILGPU.Index, ArrayView<float>, ArrayView<float>>(K_Add_Bias);
            ActivateSigmoid = accelerator.LoadAutoGroupedStreamKernel<ILGPU.Index, ArrayView<float>>(K_Activate_Sigmoid);
            DeriveSigmoid = accelerator.LoadAutoGroupedStreamKernel<ILGPU.Index, ArrayView<float>>(K_Derive_Sigmoid);
            Transpose2D = accelerator.LoadAutoGroupedStreamKernel<ILGPU.Index, ArrayView<float>, ArrayView<float>, int, int>(K_Transpose_2D);
            MultiplyErrors = accelerator.LoadAutoGroupedStreamKernel<ILGPU.Index, ArrayView<float>, ArrayView<float>>(K_Multiply_Errors);
            TransposedMatrixMultiply = accelerator.LoadAutoGroupedStreamKernel<ILGPU.Index, ArrayView<float>, ArrayView<float>, ArrayView<float>, int, int>(K_Transposed_Matrix_Multiply);
            RowSums = accelerator.LoadAutoGroupedStreamKernel<ILGPU.Index, ArrayView<float>, ArrayView<float>, int>(K_Row_Sum);
            AdjustMomentum = accelerator.LoadAutoGroupedStreamKernel<ILGPU.Index, ArrayView<float>, ArrayView<float>, float>(K_Adjust_Momentum);
            ApplyGradient = accelerator.LoadAutoGroupedStreamKernel<ILGPU.Index, ArrayView<float>, ArrayView<float>, float>(K_Apply_Gradient);
        }

        private void K_Matrix_Multiply(
            ILGPU.Index index,
            ArrayView<float> output,
            ArrayView<float> input,
            ArrayView<float> weights,
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
            output[index] = sum;
        }

        private void K_Add_Bias(ILGPU.Index index, ArrayView<float> values, ArrayView<float> additions)
        {
            int bias_index = index % additions.Length;
            values[index] = values[index] + additions[bias_index];
        }

        private void K_Activate_Sigmoid(ILGPU.Index index, ArrayView<float> values)
        {
            values[index] = 1.0f / (1.0f + XMath.Exp(-values[index]));
        }

        private void K_Derive_Sigmoid(ILGPU.Index index, ArrayView<float> values)
        {
            float item = values[index];
            values[index] = item * (1.0f - item);
        }

        private void K_Transpose_2D(ILGPU.Index index, ArrayView<float> result, ArrayView<float> values, int rows, int cols)
        {
            int offset = index * rows;
            for (int i = 0; i < rows; i++)
            {
                result[offset + i] = values[index + i * cols];
            }
        }

        private void K_Multiply_Errors(ILGPU.Index index, ArrayView<float> values, ArrayView<float> errors)
        {
            values[index] = values[index] * errors[index];
        }

        private void K_Transposed_Matrix_Multiply(
            ILGPU.Index index,
            ArrayView<float> output,
            ArrayView<float> input,
            ArrayView<float> errors,
            int input_columns,
            int out_columns
        )
        {
            int outRow = index / out_columns;
            int outColumn = index % out_columns;
            float sum = 0f;
            for (int c = 0; c < out_columns; c++)
            {
                int leftIndex = outRow + (input_columns * c);
                int rightIndex = outColumn * out_columns + c;
                sum += input[leftIndex] * errors[rightIndex];
            }
            output[index] = sum;
        }

        private void K_Row_Sum(ILGPU.Index index, ArrayView<float> sums, ArrayView<float> values, int columns)
        {
            float sum = 0f;
            for (int i = 0; i < columns; i++)
            {
                int idx = columns * index + i;
                sum += values[idx];
            }
            sums[index] = sum;
        }

        private void K_Adjust_Momentum(ILGPU.Index index, ArrayView<float> momentum, ArrayView<float> errors, float batchMultiple)
        {
            momentum[index] = momentum[index] * 0.9f + ((errors[index] * 1.0f) * batchMultiple);
        }

        private void K_Apply_Gradient(ILGPU.Index index, ArrayView<float> weights, ArrayView<float> gradient, float learning_rate)
        {
            weights[index] = weights[index] - (gradient[index] * learning_rate);
        }
    }
}
