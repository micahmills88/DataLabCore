using ILGPU;
using System;
using System.Collections.Generic;
using System.Text;

namespace DataLabCore
{
    public partial class TensorKernels
    {
        static void K_Forward_Correlation(
            ILGPU.Index index,
            ArrayView<float> output,
            ArrayView<float> left_values,
            ArrayView<float> right_values,
            int left_rows,
            int left_cols,
            int left_lays,
            int right_rows,
            int right_cols,
            int right_lays,
            int right_cubes
        )
        {
            int out_rows = left_rows - (right_rows - 1);
            int out_cols = left_cols - (right_cols - 1);
            int out_lays = right_cubes;
            int outLayerSize = out_rows * out_cols;
            int outCubeSize = outLayerSize * out_lays;

            int left_layersize = left_rows * left_cols;
            int left_cubesize = left_layersize * left_lays;

            int right_layersize = right_rows * right_cols;
            int right_cubesize = right_layersize * right_lays;

            //output indices
            int outCubeIndex = index / outCubeSize;
            int outLayerIndex = (index % outCubeSize) / outLayerSize;
            int outRowIndex = (index % outLayerSize) / out_cols;
            int outColIndex = index % out_cols;
            //source indices
            int sourceCubeIndex = outCubeIndex * left_cubesize;
            //filter indices
            int filterCubeIndex = outLayerIndex * right_cubesize;

            float sum = 0f;
            for (int l = 0; l < right_lays; l++) //foreach layer of the filter (should match input layers per cube)
            {
                int sourceLayerIndex = sourceCubeIndex + (l * left_layersize);
                int filterLayerIndex = filterCubeIndex + (l * right_layersize);
                for (int r = 0; r < right_rows; r++)
                {
                    int sourceRowIndex = sourceLayerIndex + (outRowIndex * left_cols) + (r * left_cols);
                    int filterRowIndex = filterLayerIndex + (r * right_cols);
                    for (int c = 0; c < right_cols; c++)
                    {
                        int sourceIndex = sourceRowIndex + outColIndex + c;
                        int filterIndex = filterRowIndex + c;
                        sum += left_values[sourceIndex] * right_values[filterIndex];
                    }
                }
            }
            output[index] = sum;
        }

        static void K_Input_Error_Convolution(
            ILGPU.Index index,
            ArrayView<float> output,
            ArrayView<float> left_values,
            ArrayView<float> right_values,
            int left_rows,
            int left_cols,
            int left_lays,
            int right_rows,
            int right_cols,
            int right_lays,
            int right_cubes
        )
        {
            int out_rows = left_rows - (right_rows - 1);
            int out_columns = left_cols - (right_cols - 1);
            int outLayerSize = out_rows * out_columns;
            int outCubeSize = outLayerSize * right_lays;

            int left_layersize = left_rows * left_cols;
            int left_cubesize = left_layersize * left_lays;
            int right_layersize = right_rows * right_cols;
            int right_cubesize = right_layersize * right_lays;

            int sourceCube = (index / outCubeSize) * left_cubesize;
            int sourceRowStart = ((index % outLayerSize) / out_columns) * left_cols;
            int sourceColumnStart = index % out_columns;
            int filterLayerStart = ((index % outCubeSize) / outLayerSize) * right_layersize;

            float sum = 0f;
            for (int q = 0; q < right_cubes; q++)
            {
                int sourceLayer = sourceCube + (q * left_layersize) + sourceRowStart;
                int filterLayer = filterLayerStart + (q * right_cubesize);
                for (int r = 0; r < right_rows; r++)
                {
                    int sourceRow = sourceLayer + (r * left_cols) + sourceColumnStart;
                    int filterRow = filterLayer + (r * right_cols);
                    for (int c = 0; c < right_cols; c++)
                    {
                        int sourceIndex = sourceRow + c;
                        int filterIndex = filterRow + c;
                        sum += left_values[sourceIndex] * right_values[filterIndex];
                    }
                }
                output[index] = sum;
            }
        }

        static void K_Weight_Error_Correlation(
            ILGPU.Index index,
            ArrayView<float> output,
            ArrayView<float> left_values, //left is layer inputs
            ArrayView<float> right_values, //right is output_errors
            int left_rows,
            int left_cols,
            int left_lays,
            int left_cubes,
            int right_rows,
            int right_cols,
            int right_lays
        )
        {
            int out_rows = left_rows - (right_rows - 1);
            int out_columns = left_cols - (right_cols - 1);
            int outLayerSize = out_rows * out_columns;
            int outCubeSize = outLayerSize * left_lays;

            int left_layersize = left_rows * left_cols;
            int left_cubesize = left_layersize * left_lays;
            int right_layersize = right_rows * right_cols;
            int right_cubesize = right_layersize * right_lays;

            int outCubeIndex = index / outCubeSize;
            int outLayerIndex = (index % outCubeSize) / outLayerSize;
            int outRowIndex = (index % outLayerSize) / out_columns;
            int outColIndex = index % out_columns;
            int sourceLayerIndex = outLayerIndex * left_layersize;
            int errorLayerIndex = outCubeIndex * right_layersize;

            float sum = 0f;
            for (int q = 0; q < left_cubes; q++)
            {
                int sourceCubeIndex = (q * left_cubesize) + sourceLayerIndex;
                int errorCubeIndex = (q * right_cubesize) + errorLayerIndex;
                for (int r = 0; r < right_rows; r++)
                {
                    int sourceRowIndex = sourceCubeIndex + (outRowIndex * left_cols) + (r * left_cols);
                    int errorRowIndex = errorCubeIndex + (r * right_cols);
                    for (int c = 0; c < right_cols; c++)
                    {
                        int sourceIndex = sourceRowIndex + outColIndex + c;
                        int errorIndex = errorRowIndex + c;
                        sum += left_values[sourceIndex] * right_values[errorIndex];
                    }
                }
            }
            output[index] = sum;
        }

            static void K_Activate_ReLU(ILGPU.Index index, ArrayView<float> values)
        {
            //leaky relu
            float value = values[index];
            if(value <= 0f)
            {
                values[index] = 0.1f * value;
            }
        }

        static void K_Derive_ReLU(ILGPU.Index index, ArrayView<float> results, ArrayView<float> values)
        {
            //leaky relu
            float result = 1.0f;
            if (values[index] <= 0f)
            {
                result = 0.1f;
            }
            results[index] = result;
        }

        static void K_Pad(ILGPU.Index index, ArrayView<float> result, ArrayView<float> values, int val_cols, int val_rows, int xpad, int ypad)
        {
            //the number of threads will be total rows (rows * layers * cubes)
            int destLayer = index / val_rows;
            int destRow = index % val_cols;

            int destRowSize = (xpad * 2) + val_cols;
            int paddingSize = destRowSize * ypad;
            int layerSize = (destRowSize * val_rows) + (paddingSize * 2);

            int startIndex = paddingSize + (destLayer * layerSize) + ypad;

            int destIdx = startIndex + (destRow * destRowSize);
            int srcIdx = index * val_cols;
            for (int i = 0; i < val_cols; i++)
            {
                result[destIdx + i] = values[srcIdx + i];
            }
        }

        static void K_Invert_Filter(ILGPU.Index index, ArrayView<float> inverted, ArrayView<float> normal, int filterLayerSize)
        {
            //index is number of filter layers
            int idx = index * filterLayerSize;
            for (int i = 0; i < filterLayerSize; i++)
            {
                int srcIdx = idx + i;
                int dstIdx = idx + (filterLayerSize - 1 - i);
                inverted[dstIdx] = normal[srcIdx];
            }
        }

        static void K_Max_Pool_Forward(
            ILGPU.Index index,          //total number of outputs
            ArrayView<float> mask,      //same size as inputs
            ArrayView<float> outputs,   //1/4th the size of inputs
            ArrayView<float> inputs,    //same size as mask
            int out_cols,               //out cols is half the in cols
            int in_cols
        )
        {
            //read 4 values from the inputs
            int outRow = index / out_cols;
            int outCol = index % out_cols;

            int inputOffset = (outRow * (in_cols * 2)) + (outCol * 2);

            float in_0 = inputs[inputOffset];
            float in_1 = inputs[inputOffset + 1];
            float in_2 = inputs[inputOffset + in_cols];
            float in_3 = inputs[inputOffset + in_cols + 1];

            int maxIndexOffset = inputOffset;
            float max = in_0;
            if(in_1 > max)
            {
                max = in_1;
                maxIndexOffset = inputOffset + 1;
            }
            if(in_2 > max)
            {
                max = in_2;
                maxIndexOffset = inputOffset + in_cols;
            }
            if (in_3 > max)
            {
                max = in_3;
                maxIndexOffset = inputOffset + in_cols + 1;
            }

            mask[maxIndexOffset] = 1f;
            outputs[index] = max;
        }

        static void K_Max_Pool_Backward(
            ILGPU.Index index,          //same as error size
            ArrayView<float> mask,      //same size as values
            ArrayView<float> values,    //values to be returned
            ArrayView<float> errors,
            int err_cols,
            int val_cols
        )
        {
            //read 4 values from the inputs
            int errRow = index / err_cols;
            int errCol = index % err_cols;

            int valueOffset = (errRow * (val_cols * 2)) + (errCol * 2);

            values[valueOffset]                 = errors[index] * mask[valueOffset];
            values[valueOffset + 1]             = errors[index] * mask[valueOffset + 1];
            values[valueOffset + val_cols]      = errors[index] * mask[valueOffset + val_cols];
            values[valueOffset + val_cols + 1]  = errors[index] * mask[valueOffset + val_cols + 1];
        }

        static void K_Sum_Cubes(ILGPU.Index index, ArrayView<float> sums, ArrayView<float> values, int cubeCount, int cubeSize)
        {
            //index is same as cubesize
            float sum = 0f;
            for (int i = 0; i < cubeCount; i++)
            {
                int idx = i * cubeSize + index;
                sum += values[i];
            }
            sums[index] = sum;
        }
    }
}
