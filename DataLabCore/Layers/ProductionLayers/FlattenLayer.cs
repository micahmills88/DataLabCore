using System;
using System.Collections.Generic;
using System.Text;

namespace DataLabCore
{
    public class FlattenLayer : IProductionLayer
    {
        TensorController _controller;
        LayerConfig _config;

        int _input_rows;
        int _input_columns;
        int _input_layers;
        int _input_cubes;
        int _output_size;

        public int OutputHeight { get => _input_cubes; }
        public int OutputWidth { get => _output_size; }
        public int OutputDepth { get => 1; }
        public int OutputSize { get => _output_size; }

        Tensor _forward;
        Tensor _backward;

        public FlattenLayer(TensorController tc, int inputHeight, int inputWidth, int inputDepth, int batchSize, LayerConfig layerConfig)
        {
            _controller = tc;
            _config = layerConfig;

            _input_rows = inputHeight;
            _input_columns = inputWidth;
            _input_layers = inputDepth;
            _input_cubes = batchSize;

            _output_size = inputDepth * inputHeight * inputWidth;
            var fullsize = _output_size * batchSize;
            _forward = new Tensor(tc, batchSize, _output_size, new float[fullsize]);
        }

        public Tensor Forward(Tensor data)
        {
            _forward.CopyDataFrom(data);
            return _forward;
        }
    }
}
