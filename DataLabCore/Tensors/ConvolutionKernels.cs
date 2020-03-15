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
            for (int l = 0; l < right_lays; l++)
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
    }
}
