using System;
using System.Collections.Generic;
using System.Text;

namespace DataLabCore
{
    public class FlattenLayer : IModelLayer
    {
        int _input_rows;
        int _input_columns;
        int _input_layers;
        int _input_cubes;
        int _output_size;

        public int OutSize { get => _output_size; }

        Tensor _forward;
        Tensor _backward;

        public FlattenLayer(int inputHeight, int inputWidth, int inputDepth, int batchSize)
        {
            _input_rows = inputHeight;
            _input_columns = inputWidth;
            _input_layers = inputDepth;
            _input_cubes = batchSize;

            _output_size = inputDepth * inputHeight * inputWidth;
        }

        public Tensor Forward(Tensor data)
        {
            _forward = data;
            _backward?.SetDimensions(data.Cubes, data.CubeSize, 1, 1);
            data.SetDimensions(data.Cubes, data.CubeSize, 1, 1);
            return data;
        }

        public Tensor Backward(Tensor data, float learningRate, bool errorsNeeded = false)
        {
            _backward = data;
            _forward?.SetDimensions(_input_rows, _input_columns, _input_layers, _input_cubes);
            data.SetDimensions(_input_rows, _input_columns, _input_layers, _input_cubes);
            return data;
        }
    }
}
