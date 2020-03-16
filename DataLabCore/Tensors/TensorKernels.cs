using ILGPU;
using ILGPU.Algorithms;
using ILGPU.Runtime;
using System;
using System.Collections.Generic;
using System.Text;

namespace DataLabCore
{
    public partial class TensorKernels
    {
        public Action<ILGPU.Index, ArrayView<float>, ArrayView<float>, ArrayView<float>, int, int> MatrixMultiply;
        public Action<ILGPU.Index, ArrayView<float>, ArrayView<float>> AddBias;
        public Action<ILGPU.Index, ArrayView<float>> ActivateSigmoid;
        public Action<ILGPU.Index, ArrayView<float>, int> ActivateSoftmax;
        public Action<ILGPU.Index, ArrayView<float>> DeriveSigmoid;
        public Action<ILGPU.Index, ArrayView<float>> DeriveSoftmax;
        public Action<ILGPU.Index, ArrayView<float>, ArrayView<float>, int, int> Transpose2D;
        public Action<ILGPU.Index, ArrayView<float>, ArrayView<float>> MultiplyErrors;
        public Action<ILGPU.Index, ArrayView<float>, ArrayView<float>, ArrayView<float>, int, int, int> TransposedMatrixMultiply;
        public Action<ILGPU.Index, ArrayView<float>, ArrayView<float>, int> RowSums;
        public Action<ILGPU.Index, ArrayView<float>, ArrayView<float>, float> AdjustMomentum;
        public Action<ILGPU.Index, ArrayView<float>, ArrayView<float>, float> ApplyGradient;
        public Action<ILGPU.Index, ArrayView<float>, ArrayView<float>, ArrayView<float>, int, int> SubtractTransposed;
        public Action<ILGPU.Index, ArrayView<float>, ArrayView<float>> SUM;
        public Action<ILGPU.Index, ArrayView<float>, ArrayView<float>, ArrayView<float>> LogisticLoss;
        public Action<ILGPU.Index, ArrayView<float>, ArrayView<float>, ArrayView<float>> MultiClassLoss;
        public Action<ILGPU.Index, ArrayView<float>, ArrayView<float>, ArrayView<float>> MeanSquaredError;

        public Action<ILGPU.Index, ArrayView<float>, ArrayView<float>, ArrayView<float>, int, int, int, int, int, int, int> ForwardCorrelation;
        public Action<ILGPU.Index, ArrayView<float>> ActivateReLU;
        public Action<ILGPU.Index, ArrayView<float>, ArrayView<float>> DeriveReLU;
        public Action<ILGPU.Index, ArrayView<float>, ArrayView<float>, int, int, int, int> Pad;
        public Action<ILGPU.Index, ArrayView<float>, ArrayView<float>, int> InvertFilter;
        public Action<ILGPU.Index, ArrayView<float>, ArrayView<float>, ArrayView<float>, int, int, int, int, int, int, int> InputErrorConvolution;
        public Action<ILGPU.Index, ArrayView<float>, ArrayView<float>, ArrayView<float>, int, int, int, int, int, int, int> WeightErrorCorrelation;

        public TensorKernels(Accelerator accelerator)
        {
            MatrixMultiply = accelerator.LoadAutoGroupedStreamKernel<ILGPU.Index, ArrayView<float>, ArrayView<float>, ArrayView<float>, int, int>(K_Matrix_Multiply);
            AddBias = accelerator.LoadAutoGroupedStreamKernel<ILGPU.Index, ArrayView<float>, ArrayView<float>>(K_Add_Bias);
            ActivateSigmoid = accelerator.LoadAutoGroupedStreamKernel<ILGPU.Index, ArrayView<float>>(K_Activate_Sigmoid);
            ActivateSoftmax = accelerator.LoadAutoGroupedStreamKernel<ILGPU.Index, ArrayView<float>, int>(K_Activate_Softmax);
            DeriveSigmoid = accelerator.LoadAutoGroupedStreamKernel<ILGPU.Index, ArrayView<float>>(K_Derive_Sigmoid);
            DeriveSoftmax = accelerator.LoadAutoGroupedStreamKernel<ILGPU.Index, ArrayView<float>>(K_Derive_Softmax);
            Transpose2D = accelerator.LoadAutoGroupedStreamKernel<ILGPU.Index, ArrayView<float>, ArrayView<float>, int, int>(K_Transpose_2D);
            MultiplyErrors = accelerator.LoadAutoGroupedStreamKernel<ILGPU.Index, ArrayView<float>, ArrayView<float>>(K_Multiply_Errors);
            TransposedMatrixMultiply = accelerator.LoadAutoGroupedStreamKernel<ILGPU.Index, ArrayView<float>, ArrayView<float>, ArrayView<float>, int, int, int>(K_Transposed_Matrix_Multiply);
            RowSums = accelerator.LoadAutoGroupedStreamKernel<ILGPU.Index, ArrayView<float>, ArrayView<float>, int>(K_Row_Sum);
            AdjustMomentum = accelerator.LoadAutoGroupedStreamKernel<ILGPU.Index, ArrayView<float>, ArrayView<float>, float>(K_Adjust_Momentum);
            ApplyGradient = accelerator.LoadAutoGroupedStreamKernel<ILGPU.Index, ArrayView<float>, ArrayView<float>, float>(K_Apply_Gradient);
            SubtractTransposed = accelerator.LoadAutoGroupedStreamKernel<ILGPU.Index, ArrayView<float>, ArrayView<float>, ArrayView<float>, int, int>(K_Subtract_Transposed);
            SUM = accelerator.LoadAutoGroupedStreamKernel<ILGPU.Index, ArrayView<float>, ArrayView<float>>(K_Sum);
            LogisticLoss = accelerator.LoadAutoGroupedStreamKernel<ILGPU.Index, ArrayView<float>, ArrayView<float>, ArrayView<float>>(K_Logistic_Loss);
            MultiClassLoss = accelerator.LoadAutoGroupedStreamKernel<ILGPU.Index, ArrayView<float>, ArrayView<float>, ArrayView<float>>(K_Multi_Class_Loss);
            MeanSquaredError = accelerator.LoadAutoGroupedStreamKernel<ILGPU.Index, ArrayView<float>, ArrayView<float>, ArrayView<float>>(K_Mean_Squared_Error);
            
            ForwardCorrelation = accelerator.LoadAutoGroupedStreamKernel<ILGPU.Index, ArrayView<float>, ArrayView<float>, ArrayView<float>, int, int, int, int, int, int, int>(K_Forward_Correlation);
            ActivateReLU = accelerator.LoadAutoGroupedStreamKernel<ILGPU.Index, ArrayView<float>>(K_Activate_ReLU);
            DeriveReLU = accelerator.LoadAutoGroupedStreamKernel<ILGPU.Index, ArrayView<float>, ArrayView<float>>(K_Derive_ReLU);
            Pad = accelerator.LoadAutoGroupedStreamKernel<ILGPU.Index, ArrayView<float>, ArrayView<float>, int, int, int, int>(K_Pad);
            InvertFilter = accelerator.LoadAutoGroupedStreamKernel<ILGPU.Index, ArrayView<float>, ArrayView<float>, int>(K_Invert_Filter);
            InputErrorConvolution = accelerator.LoadAutoGroupedStreamKernel<ILGPU.Index, ArrayView<float>, ArrayView<float>, ArrayView<float>, int, int, int, int, int, int, int>(K_Input_Error_Convolution);
            WeightErrorCorrelation = accelerator.LoadAutoGroupedStreamKernel<ILGPU.Index, ArrayView<float>, ArrayView<float>, ArrayView<float>, int, int, int, int, int, int, int>(K_Weight_Error_Correlation);
        }

        static void K_Matrix_Multiply(
            ILGPU.Index index,
            ArrayView<float> output,
            ArrayView<float> left_values,
            ArrayView<float> right_values,
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
                sum += left_values[leftIndex] * right_values[rightIndex];
            }
            output[index] = sum;
        }

        static void K_Add_Bias(ILGPU.Index index, ArrayView<float> values, ArrayView<float> additions)
        {
            int bias_index = index % additions.Length;
            values[index] = values[index] + additions[bias_index];
        }

        static void K_Activate_Sigmoid(ILGPU.Index index, ArrayView<float> values)
        {
            values[index] = 1.0f / (1.0f + XMath.Exp(-values[index]));
        }

        static void K_Derive_Sigmoid(ILGPU.Index index, ArrayView<float> values)
        {
            float item = values[index];
            values[index] = item * (1.0f - item);
        }

        static void K_Derive_Softmax(ILGPU.Index index, ArrayView<float> values)
        {
            values[index] = 1f;
        }

        static void K_Activate_Softmax(ILGPU.Index index, ArrayView<float> values, int columns)
        {
            int start = index * columns;
            float rowSum = 0f;
            for (int i = 0; i < columns; i++)
            {
                float value = XMath.Exp(values[start + i]);
                rowSum += value;
                values[start + i] = value;
            }
            for (int i = 0; i < columns; i++)
            {
                values[start + i] /= rowSum;
            }
        }

        static void K_Transpose_2D(ILGPU.Index index, ArrayView<float> result, ArrayView<float> values, int rows, int cols)
        { //2*10
            int sourceRow = index % rows;
            int sourceCol = index / rows;
            int srcIndex = sourceRow * cols + sourceCol;
            result[index] = values[srcIndex];
        }

        static void K_Multiply_Errors(ILGPU.Index index, ArrayView<float> values, ArrayView<float> errors)
        {
            values[index] = values[index] * errors[index];
        }

        static void K_Transposed_Matrix_Multiply(
            ILGPU.Index index, //30
            ArrayView<float> output, //30
            ArrayView<float> input, //1*3
            ArrayView<float> errors, //10*1
            int out_columns, //same as error rowcount //10
            int err_columns, //same as error colcount //1
            int input_columns //same as input colcount //3
        )
        {
            //output dimensions should be 3*10
            int outRow = index / out_columns;
            int outColumn = index % out_columns;
            float sum = 0f;
            for (int c = 0; c < err_columns; c++)
            {
                int leftIndex = outRow + (input_columns * c);
                int rightIndex = outColumn * err_columns + c;
                sum += input[leftIndex] * errors[rightIndex];
            }
            output[index] = sum;
        }

        static void K_Row_Sum(ILGPU.Index index, ArrayView<float> sums, ArrayView<float> values, int columns)
        {
            float sum = 0f;
            for (int i = 0; i < columns; i++)
            {
                int idx = columns * index + i;
                sum += values[idx];
            }
            sums[index] = sum;
        }

        static void K_Adjust_Momentum(ILGPU.Index index, ArrayView<float> momentum, ArrayView<float> errors, float batchMultiple)
        {
            momentum[index] = (momentum[index] * 0.9f) + (0.1f *(errors[index] * batchMultiple));
        }

        static void K_Apply_Gradient(ILGPU.Index index, ArrayView<float> weights, ArrayView<float> gradient, float learning_rate)
        {
            weights[index] = weights[index] - (gradient[index] * learning_rate);
        }

        static void K_Subtract_Transposed(ILGPU.Index index, ArrayView<float> outputs, ArrayView<float> data, ArrayView<float> labels, int rows, int cols)
        {
            int offset = index * rows;
            for (int i = 0; i < rows; i++)
            {
                outputs[offset + i] = data[index + i * cols] - labels[index + i * cols];
            }
        }

        static void K_Sum(ILGPU.Index index, ArrayView<float> sums, ArrayView<float> values)
        {
            sums[index] = sums[index] + values[index];   
        }

        static void K_Logistic_Loss(ILGPU.Index index, ArrayView<float> results, ArrayView<float> values, ArrayView<float> expecteds)
        {
            //expected should be either zero or 1
            float expected = expecteds[index];
            float value = values[index];
            float result = -1.0f * ((expected * (float)Math.Log(value)) + ((1f - expected) * (float)Math.Log(1f - value)));
            results[index] += result;
        }

        static void K_Multi_Class_Loss(ILGPU.Index index, ArrayView<float> results, ArrayView<float> values, ArrayView<float> expecteds)
        {
            float expected = expecteds[index];
            float value = values[index];
            float result =  -1.0f * XMath.Log(value) * expected;
            results[index] += result;
        }

        static void K_Mean_Squared_Error(ILGPU.Index index, ArrayView<float> results, ArrayView<float> values, ArrayView<float> expecteds)
        {
            float expected = expecteds[index];
            float value = values[index];
            float result = (value - expected) * (value - expected);
            results[index] += result;
        }
    }
}
