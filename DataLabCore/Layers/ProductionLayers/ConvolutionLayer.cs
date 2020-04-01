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

        public int OutHeight { get => _output_height; }
        public int OutWidth { get => _output_width; }
        public int OutDepth { get => _output_depth; }

        int _batch_size;
        ActivationType _activation;
        PaddingType _padding;

        Tensor _filter_weights;
        Tensor _filter_bias;
        Tensor _inputs;
        Tensor _outputs;

        public ConvolutionLayer(
            TensorController controller,
            float[] weights,
            float[] bias,
            int inputHeight, int inputWidth, int inputDepth,
            int filterHeight, int filterWidth, int filterCount,
            int batchSize,
            ActivationType layerActivation,
            PaddingType paddingType = PaddingType.None
            )
        {
            _controller = controller;
            _filter_height = filterHeight;
            _filter_width = filterWidth;
            _filter_depth = inputDepth;
            _filter_count = filterCount;
            _activation = layerActivation;
            _output_depth = filterCount;
            _batch_size = batchSize;

            _activation = layerActivation;
            _padding = paddingType;

            if(_padding == PaddingType.Same)
            {
                _output_height = inputHeight;
                _output_width = inputWidth;
            }
            else
            {
                _output_height = inputHeight - (filterHeight - 1);
                _output_width = inputWidth - (filterWidth - 1);
            }

            _filter_weights = new Tensor(_controller, _filter_height, _filter_width, _filter_depth, _filter_count, weights);
            _filter_bias = new Tensor(_controller, _output_height, _output_width, _output_depth, bias);

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
