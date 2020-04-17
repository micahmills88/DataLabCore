using System;
using System.Collections.Generic;
using System.Text;

namespace DataLabCore
{
    public class MaxPoolLayer : IProductionLayer
    {
        TensorController _controller;
        LayerConfig _config;

        int _input_height;
        int _input_width;
        int _input_depth;
        int _batch_size;

        int _output_height;
        int _output_width;
        int _output_depth;

        public int OutputHeight { get => _output_height; }
        public int OutputWidth { get => _output_width; }
        public int OutputDepth { get => _output_depth; }
        public int OutputSize { get => _output_height * _output_width * _output_depth; }

        Tensor _outputs;
        Tensor _mask;

        public MaxPoolLayer(TensorController tc, int inputHeight, int inputWidth, int inputDepth, int batchSize, LayerConfig layerConfig)
        {
            _controller = tc;
            _config = layerConfig;

            _input_height = inputHeight;
            _input_width = inputWidth;
            _input_depth = inputDepth;
            _batch_size = batchSize;

            //TODO this layer will break if the input dimensions are odd
            //need to add padding functionality for odd layers
            _output_height = _input_height / 2;
            _output_width = _input_width / 2;
            _output_depth = _input_depth;

            var out_size = _output_height * _output_width * _output_depth * _batch_size;
            _outputs = new Tensor(tc, _output_height, _output_width, _output_depth, _batch_size, new float[out_size]);

            var input_size = _input_height * _input_width * _input_depth * _batch_size;
            _mask = new Tensor(tc, _input_height, _input_width, _input_depth, _batch_size, new float[input_size]);

        }

        public Tensor Forward(Tensor data)
        {
            _controller.MaxPoolForward(data, _outputs, _mask);
            return _outputs;
        }
    }
}
