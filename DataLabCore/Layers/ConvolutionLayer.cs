using System;
using System.Collections.Generic;
using System.Text;

namespace DataLabCore
{
    public class ConvolutionLayer : IModelLayer
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

        public ConvolutionLayer(TensorController controller, int inputHeight, int inputWidth, int inputDepth, int filterHeight, int filterWidth, int filterCount, int batchSize, ActivationType layerActivation)
        {
            _controller = controller;
            _filter_height = filterHeight;
            _filter_width = filterWidth;
            _filter_depth = inputDepth;
            _filter_count = filterCount;
            _output_height = inputHeight - (filterHeight - 1);
            _output_width = inputWidth - (filterWidth - 1);
            _output_depth = filterCount;
            _batch_size = batchSize;
            _activation = layerActivation;

            float weight_range = (float)Math.Sqrt(2.0d / (float)(inputHeight * inputWidth * inputDepth));

            int filterSize = _filter_height * _filter_width * _filter_depth * _filter_count;
            var weight_data = RandomGenerator.GetFloatDistribution(filterSize, 0f, weight_range);
            _filter_weights = new Tensor(_controller, _filter_height, _filter_width, _filter_depth, _filter_count, weight_data);
            _momentum_filter_weights = new Tensor(_controller, _filter_height, _filter_width, _filter_depth, _filter_count, new float[filterSize]);
            _filter_weights_errors = new Tensor(_controller, _filter_height, _filter_width, _filter_depth, _filter_count, new float[filterSize]);
            _inverted_filters = new Tensor(_controller, _filter_height, _filter_width, _filter_depth, _filter_count, new float[filterSize]);

            int biasSize = _output_height * _output_width * _output_depth;
            float[] filter_bias = RandomGenerator.GetFloatDistribution(biasSize, 0f, weight_range);
            _filter_bias = new Tensor(_controller, _output_height, _output_width, _output_depth, filter_bias);
            _filter_bias_errors = new Tensor(_controller, _output_height, _output_width, _output_depth, new float[biasSize]);
            _momentum_filter_bias = new Tensor(_controller, _output_height, _output_width, _output_depth, new float[biasSize]);

            var output_size = _output_height * _output_width * _output_depth * batchSize;
            _outputs = new Tensor(_controller, _output_height, _output_width, _output_depth, batchSize, new float[output_size]);
            _output_errors = new Tensor(_controller, _output_height, _output_width, _output_depth, batchSize, new float[output_size]);

            var input_size = inputHeight * inputWidth * inputDepth * batchSize;
            _input_errors = new Tensor(_controller, inputHeight, inputWidth, inputDepth, batchSize, new float[input_size]);

            var xpad = filterWidth - 1;
            var ypad = filterHeight - 1;
            var padded_width = _output_width + (xpad * 2);
            var padded_height = _output_height + (ypad * 2);
            var padded_size = padded_height * padded_width * _output_depth * batchSize;
            _padded_errors = new Tensor(_controller, padded_width, padded_height, _output_depth, batchSize, new float[padded_size]);
        }

        public Tensor Forward(Tensor input)
        {
            _inputs = input;
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
