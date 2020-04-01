using System;
using System.Collections.Generic;
using System.Text;

namespace DataLabCore
{
    public class ConvolutionLayerT : ITrainableLayer
    {
        TensorController _controller;

        int _filter_height;
        int _filter_width;
        int _filter_depth;
        int _filter_count;

        int _output_height;
        int _output_width;
        int _output_depth;

        public int OutHeight { get => _output_height;}
        public int OutWidth { get => _output_width; }
        public int OutDepth { get => _output_depth; }

        int _batch_size;
        ActivationType _activation;
        PaddingType _padding;

        Tensor _filter_weights;
        Tensor _filter_bias;

        Tensor _filter_weights_errors;
        Tensor _filter_bias_errors;

        Tensor _momentum_filter_weights;
        Tensor _momentum_filter_bias;

        Tensor _inputs;
        Tensor _outputs;

        Tensor _input_errors;
        Tensor _output_errors;

        Tensor _padded_errors;
        Tensor _inverted_filters;

        public ConvolutionLayerT(
            TensorController controller, 
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
            _batch_size = batchSize;
            _activation = layerActivation;
            _padding = paddingType;
            _output_depth = filterCount;


            var xpad = filterWidth - 1;
            var ypad = filterHeight - 1;
            var padded_width = 0;
            var padded_height = 0;
            if (_padding == PaddingType.Same)
            {
                _output_height = inputHeight;
                _output_width = inputWidth;
                var padInputHeight = inputHeight + ypad;
                var padInputWidth = inputWidth + xpad;
                var padInputSize = padInputHeight * padInputWidth * inputDepth * batchSize;
                _inputs = new Tensor(_controller, padInputHeight, padInputWidth, inputDepth, batchSize, new float[padInputSize]);

                padded_width = _output_width + (xpad * 1);
                padded_height = _output_height + (ypad * 1);
            }
            else //PaddingType.None
            {
                _output_height = inputHeight - ypad;
                _output_width = inputWidth - xpad;

                padded_width = _output_width + (xpad * 2);
                padded_height = _output_height + (ypad * 2);
            }
            var padded_size = padded_height * padded_width * _output_depth * batchSize;
            _padded_errors = new Tensor(_controller, padded_height, padded_width, _output_depth, batchSize, new float[padded_size]);

            //he_normal for relu
            float weight_range = (float)Math.Sqrt(2.0d / (float)(filterHeight * filterWidth * _filter_depth));
            //float weight_range = (float)Math.Sqrt(2.0d / (float)(inputHeight * inputWidth * _filter_depth));

            int filterSize = _filter_height * _filter_width * _filter_depth * _filter_count;
            var weight_data = RandomGenerator.GetFloatNormalDistribution(filterSize, 0f, weight_range);
            _filter_weights = new Tensor(_controller, _filter_height, _filter_width, _filter_depth, _filter_count, weight_data);
            _momentum_filter_weights = new Tensor(_controller, _filter_height, _filter_width, _filter_depth, _filter_count, new float[filterSize]);
            _filter_weights_errors = new Tensor(_controller, _filter_height, _filter_width, _filter_depth, _filter_count, new float[filterSize]);
            _inverted_filters = new Tensor(_controller, _filter_height, _filter_width, _filter_depth, _filter_count, new float[filterSize]);

            int biasSize = _output_height * _output_width * _output_depth;
            float[] filter_bias = new float[biasSize]; // RandomGenerator.GetFloatUniformDistribution(biasSize, -weight_range, weight_range);
            _filter_bias = new Tensor(_controller, _output_height, _output_width, _output_depth, filter_bias);
            _filter_bias_errors = new Tensor(_controller, _output_height, _output_width, _output_depth, new float[biasSize]);
            _momentum_filter_bias = new Tensor(_controller, _output_height, _output_width, _output_depth, new float[biasSize]);

            var output_size = _output_height * _output_width * _output_depth * batchSize;
            _outputs = new Tensor(_controller, _output_height, _output_width, _output_depth, batchSize, new float[output_size]);
            _output_errors = new Tensor(_controller, _output_height, _output_width, _output_depth, batchSize, new float[output_size]);

            var input_size = inputHeight * inputWidth * inputDepth * batchSize;
            _input_errors = new Tensor(_controller, inputHeight, inputWidth, inputDepth, batchSize, new float[input_size]);
        }

        public Tensor Forward(Tensor input)
        {
            if(_padding == PaddingType.Same)
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

        public Tensor Backward(Tensor errors, float learningRate, bool calculateInputErrors = true)
        {
            _controller.ConvolutionOutputError(_output_errors, _outputs, errors, _activation);
            if(calculateInputErrors)
            {
                _controller.ConvolutionInputError(_input_errors, _padded_errors, _output_errors, _inverted_filters, _filter_weights);
            }
            float batchMultiple = (1.0f / (float)_batch_size);
            _controller.ConvolutionLayerWeightUpdate(_filter_weights, _filter_weights_errors, _momentum_filter_weights, _inputs, _output_errors, batchMultiple, learningRate);
            _controller.ConvolutionLayerBiasUpdate(_filter_bias, _filter_bias_errors, _momentum_filter_bias, _output_errors, batchMultiple, learningRate);
            return _input_errors;
        }
    }
}
