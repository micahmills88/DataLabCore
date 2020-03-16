using System;
using System.Collections.Generic;
using System.Text;

namespace DataLabCore
{
    public class FlattenLayer : IModelLayer
    {
        int input_rows;
        int input_columns;
        int input_layers;
        int input_cubes;

        public readonly int output_size;

        public FlattenLayer(int inputHeight, int inputWidth, int inputDepth, int batchSize)
        {
            input_rows = inputHeight;
            input_columns = inputWidth;
            input_layers = inputDepth;
            input_cubes = batchSize;

            output_size = inputDepth * inputHeight * inputWidth;
        }

        public Tensor Forward(Tensor data)
        {
            data.SetDimensions(data.Cubes, data.CubeSize, 1, 1);
            return data;
        }

        public Tensor Backward(Tensor data, float learningRate, bool errorsNeeded = false)
        {
            data.SetDimensions(input_rows, input_columns, input_layers, input_cubes);
            return data;
        }
    }
}
