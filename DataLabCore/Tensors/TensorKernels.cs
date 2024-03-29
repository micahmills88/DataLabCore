﻿using ILGPU;
using ILGPU.Algorithms;
using ILGPU.Runtime;
using System;
using System.Collections.Generic;
using System.Text;

namespace DataLabCore
{
    public partial class TensorKernels
    {
        public Action<Index1D, ArrayView<float>, ArrayView<float>, ArrayView<float>, int, int> MatrixMultiply;
        public Action<Index1D, ArrayView<float>, ArrayView<float>> AddBias;
        public Action<Index1D, ArrayView<float>> ActivateSigmoid;
        public Action<Index1D, ArrayView<float>, int> ActivateSoftmax;
        public Action<Index1D, ArrayView<float>, ArrayView<float>> DeriveSigmoid;
        public Action<Index1D, ArrayView<float>> DeriveSoftmax;
        public Action<Index1D, ArrayView<float>, ArrayView<float>, int, int> Transpose2D;
        public Action<Index1D, ArrayView<float>, ArrayView<float>> MultiplyErrors;
        public Action<Index1D, ArrayView<float>, ArrayView<float>, ArrayView<float>, int, int, int> TransposedMatrixMultiply;
        public Action<Index1D, ArrayView<float>, ArrayView<float>, int> RowSums;
        public Action<Index1D, ArrayView<float>, ArrayView<float>, float> AdjustMomentum;
        public Action<Index1D, ArrayView<float>, ArrayView<float>, float> ApplyGradient;
        public Action<Index1D, ArrayView<float>, ArrayView<float>, ArrayView<float>, int, int> SubtractTransposed;
        public Action<Index1D, ArrayView<float>, ArrayView<float>> SUM;
        public Action<Index1D, ArrayView<float>, ArrayView<float>, ArrayView<float>> LogisticLoss;
        public Action<Index1D, ArrayView<float>, ArrayView<float>, ArrayView<float>> MultiClassLoss;
        public Action<Index1D, ArrayView<float>, ArrayView<float>, ArrayView<float>> MeanSquaredError;

        public Action<Index1D, ArrayView<float>, ArrayView<float>, ArrayView<float>, int, int, int, int, int, int, int> ForwardCorrelation;
        public Action<Index1D, ArrayView<float>> ActivateReLU;
        public Action<Index1D, ArrayView<float>, ArrayView<float>> DeriveReLU;
        public Action<Index1D, ArrayView<float>, ArrayView<float>, int, int, int, int> Pad;
        public Action<Index1D, ArrayView<float>, ArrayView<float>, int> InvertFilter;
        public Action<Index1D, ArrayView<float>, ArrayView<float>, ArrayView<float>, int, int, int, int, int, int, int> InputErrorConvolution;
        public Action<Index1D, ArrayView<float>, ArrayView<float>, ArrayView<float>, int, int, int, int, int, int, int> WeightErrorCorrelation;
        public Action<Index1D, ArrayView<float>, ArrayView<float>, int, int> SumCubes;

        public Action<Index1D, ArrayView<float>, ArrayView<float>, ArrayView<float>, int, int> MaxPoolForward;
        public Action<Index1D, ArrayView<float>, ArrayView<float>, ArrayView<float>, int, int> MaxPoolBackward;

        public TensorKernels(Accelerator accelerator)
        {
            MatrixMultiply = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<float>, ArrayView<float>, ArrayView<float>, int, int>(K_Matrix_Multiply);
            AddBias = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<float>, ArrayView<float>>(K_Add_Bias);
            ActivateSigmoid = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<float>>(K_Activate_Sigmoid);
            ActivateSoftmax = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<float>, int>(K_Activate_Softmax);
            DeriveSigmoid = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<float>, ArrayView<float>>(K_Derive_Sigmoid);
            DeriveSoftmax = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<float>>(K_Derive_Softmax);
            Transpose2D = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<float>, ArrayView<float>, int, int>(K_Transpose_2D);
            MultiplyErrors = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<float>, ArrayView<float>>(K_Multiply_Errors);
            TransposedMatrixMultiply = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<float>, ArrayView<float>, ArrayView<float>, int, int, int>(K_Transposed_Matrix_Multiply);
            RowSums = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<float>, ArrayView<float>, int>(K_Row_Sum);
            AdjustMomentum = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<float>, ArrayView<float>, float>(K_Adjust_Momentum);
            ApplyGradient = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<float>, ArrayView<float>, float>(K_Apply_Gradient);
            SubtractTransposed = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<float>, ArrayView<float>, ArrayView<float>, int, int>(K_Subtract_Transposed);
            SUM = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<float>, ArrayView<float>>(K_Sum);
            LogisticLoss = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<float>, ArrayView<float>, ArrayView<float>>(K_Logistic_Loss);
            MultiClassLoss = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<float>, ArrayView<float>, ArrayView<float>>(K_Multi_Class_Loss);
            MeanSquaredError = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<float>, ArrayView<float>, ArrayView<float>>(K_Mean_Squared_Error);
            
            ForwardCorrelation = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<float>, ArrayView<float>, ArrayView<float>, int, int, int, int, int, int, int>(K_Forward_Correlation);
            ActivateReLU = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<float>>(K_Activate_ReLU);
            DeriveReLU = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<float>, ArrayView<float>>(K_Derive_ReLU);
            Pad = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<float>, ArrayView<float>, int, int, int, int>(K_Pad);
            InvertFilter = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<float>, ArrayView<float>, int>(K_Invert_Filter);
            InputErrorConvolution = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<float>, ArrayView<float>, ArrayView<float>, int, int, int, int, int, int, int>(K_Input_Error_Convolution);
            WeightErrorCorrelation = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<float>, ArrayView<float>, ArrayView<float>, int, int, int, int, int, int, int>(K_Weight_Error_Correlation);
            SumCubes = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<float>, ArrayView<float>, int, int>(K_Sum_Cubes);

            MaxPoolForward = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<float>, ArrayView<float>, ArrayView<float>, int, int>(K_Max_Pool_Forward);
            MaxPoolBackward = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<float>, ArrayView<float>, ArrayView<float>, int, int>(K_Max_Pool_Backward);
        }

        static void K_Matrix_Multiply(
            Index1D index,
            ArrayView<float> output,
            ArrayView<float> left_values,
            ArrayView<float> right_values,
            int input_columns,
            int out_columns
        )
        {
            int outRow = index / out_columns;
            int outCol = index % out_columns;
            int leftOffset = outRow * input_columns;

            float sum = 0f;
            for (int c = 0; c < input_columns; c++)
            {
                int leftIdx = leftOffset + c;
                int rightIdx = outCol + (c * out_columns);
                sum += left_values[leftIdx] * right_values[rightIdx];
            }
            output[index] = sum;
        }

        static void K_Add_Bias(Index1D index, ArrayView<float> values, ArrayView<float> additions)
        {
            int bias_index = (int)(index % additions.Length);
            var val = values[index] + additions[bias_index];
            values[index] = val;
        }

        static void K_Activate_Sigmoid(Index1D index, ArrayView<float> values)
        {
            var val = 1.0f / (1.0f + XMath.Exp(-values[index]));
            values[index] = val;
        }

        static void K_Derive_Sigmoid(Index1D index, ArrayView<float> outputs, ArrayView<float> inputs)
        {
            float item = inputs[index];
            outputs[index] = item * (1.0f - item);
        }

        static void K_Derive_Softmax(Index1D index, ArrayView<float> values)
        {
            values[index] = 1f;
        }

        static void K_Activate_Softmax(Index1D index, ArrayView<float> values, int columns)
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

        static void K_Transpose_2D(Index1D index, ArrayView<float> result, ArrayView<float> values, int rows, int cols)
        { //2*10
            int sourceRow = index % rows;
            int sourceCol = index / rows;
            int srcIndex = sourceRow * cols + sourceCol;
            result[index] = values[srcIndex];
        }

        static void K_Multiply_Errors(Index1D index, ArrayView<float> values, ArrayView<float> errors)
        {
            values[index] = values[index] * errors[index];
        }

        static void K_Transposed_Matrix_Multiply(
            Index1D index, //30
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

        static void K_Row_Sum(Index1D index, ArrayView<float> sums, ArrayView<float> values, int columns)
        {
            float sum = 0f;
            int startIndex = index * columns;
            for (int i = 0; i < columns; i++)
            {
                sum += values[startIndex + i];
            }
            sums[index] = sum;
        }

        static void K_Adjust_Momentum(Index1D index, ArrayView<float> momentum, ArrayView<float> errors, float batchMultiple)
        {
            momentum[index] = (momentum[index] * 0.9f) + (0.1f *(errors[index] * batchMultiple));
        }

        static void K_Apply_Gradient(Index1D index, ArrayView<float> weights, ArrayView<float> gradient, float learning_rate)
        {
            weights[index] -= (gradient[index] * learning_rate);
        }

        static void K_Subtract_Transposed(Index1D index, ArrayView<float> outputs, ArrayView<float> data, ArrayView<float> labels, int rows, int cols)
        {
            int sourceRow = index % rows;
            int sourceCol = index / rows;
            int srcIndex = sourceRow * cols + sourceCol;
            outputs[index] = data[srcIndex] - labels[srcIndex];
        }

        static void K_Sum(Index1D index, ArrayView<float> sums, ArrayView<float> values)
        {
            sums[index] = sums[index] + values[index];   
        }

        static void K_Logistic_Loss(Index1D index, ArrayView<float> results, ArrayView<float> values, ArrayView<float> expecteds)
        {
            //expected should be either zero or 1
            float expected = expecteds[index];
            float value = values[index];
            float result = -1.0f * ((expected * (float)Math.Log(value)) + ((1f - expected) * (float)Math.Log(1f - value)));
            results[index] += result;
        }

        static void K_Multi_Class_Loss(Index1D index, ArrayView<float> results, ArrayView<float> values, ArrayView<float> expecteds)
        {
            float expected = expecteds[index];
            float value = values[index];
            float result =  -1.0f * XMath.Log(value) * expected;
            results[index] += result;
        }

        static void K_Mean_Squared_Error(Index1D index, ArrayView<float> results, ArrayView<float> values, ArrayView<float> expecteds)
        {
            float expected = expecteds[index];
            float value = values[index];
            float result = (value - expected) * (value - expected);
            results[index] += result;
        }
    }
}
