using System;
using System.Collections.Generic;
using System.Text;

namespace DataLabCore
{
    public class ConvolutionLayer : IProductionLayer
    {
        TensorController _controller;

        int _filter_height;
        int _filter_width;
        int _filter_depth;
        int _filter_count;

        int _output_height;
        int _output_width;
        int _output_depth;

        public int OutputHeight { get => _output_height; }
        public int OutputWidth { get => _output_width; }
        public int OutputDepth { get => _output_depth; }
        public int OutputSize { get => _output_height * _output_width * _output_depth; }

        int _batch_size;
        ActivationType _activation;
        PaddingType _padding;

        Tensor _filter_weights;
        Tensor _filter_bias;
        Tensor _inputs;
        Tensor _outputs;

        LayerConfig _config;

        public ConvolutionLayer(TensorController tc, int inputHeight, int inputWidth, int inputDepth, int batchSize, LayerConfig layerConfig)
        {
            _controller = tc;
            _config = layerConfig;
            _filter_height = layerConfig.WeightRows;
            _filter_width = layerConfig.WeightColumns;
            _filter_depth = inputDepth;
            _filter_count = layerConfig.WeightCubes;
            _batch_size = batchSize;
            _activation = layerConfig.activationType;
            _padding = layerConfig.paddingType;
            _output_depth = _filter_count;

            var xpad = _filter_width - 1;
            var ypad = _filter_height - 1;
            if (_padding == PaddingType.Same)
            {
                _output_height = inputHeight;
                _output_width = inputWidth;
                var padInputHeight = inputHeight + ypad;
                var padInputWidth = inputWidth + xpad;
                var padInputSize = padInputHeight * padInputWidth * inputDepth * batchSize;
                _inputs = new Tensor(_controller, padInputHeight, padInputWidth, inputDepth, batchSize, new float[padInputSize]);
            }
            else //PaddingType.None
            {
                _output_height = inputHeight - ypad;
                _output_width = inputWidth - xpad;
            }

            _filter_weights = new Tensor(_controller, _filter_height, _filter_width, _filter_depth, _filter_count, _config.Weights);
            _filter_bias = new Tensor(_controller, _output_height, _output_width, _output_depth, _config.Bias);

            var output_size = _output_height * _output_width * _output_depth * batchSize;
            _outputs = new Tensor(_controller, _output_height, _output_width, _output_depth, batchSize, new float[output_size]);
        }

        public Tensor Forward(Tensor input)
        {
            if (_padding == PaddingType.Same)
            {
                int xpad = (_filter_width - 1) / 2;
                int ypad = (_filter_height - 1) / 2;
                _controller.PadTensor(_inputs, input, xpad, ypad);
            }
            else
            {
                _inputs = input;
            }
            _controller.ConvolutionForward(_outputs, _inputs, _filter_weights, _filter_bias, _activation);
            return _outputs;
        }
    }
}
