using System;
using System.Collections.Generic;
using System.Text;

namespace DataLabCore
{
    class ConvolutionLayer : IModelLayer
    {
        TensorController _controller;

        int _filter_height;
        int _filter_width;
        int _filter_depth;
        int _filter_count;

        int _output_height;
        int _output_width;
        int _output_depth;
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

        public ConvolutionLayer(TensorController controller, int inputHeight, int inputWidth, int inputDepth, int filterHeight, int filterWidth, int filterCount, ActivationType layerActivation)
        {
            _controller = controller;
            _filter_height = filterHeight;
            _filter_width = filterWidth;
            _filter_depth = inputDepth;
            _filter_count = filterCount;
            _output_height = inputHeight - (filterHeight - 1);
            _output_width = inputWidth - (filterWidth - 1);
            _output_depth = filterCount;

            int filterSize = _filter_height * _filter_width * _filter_depth * _filter_count;
            float weight_range = (float)Math.Sqrt(1.0d / (float)filterSize);
            var weight_data = RandomGenerator.GetFloatDistribution(filterSize, 0f, weight_range);
            _filter_weights = new Tensor(_controller, _filter_height, _filter_width, _filter_depth, _filter_count, weight_data);
            _momentum_filter_weights = new Tensor(_controller, _filter_height, _filter_width, _filter_depth, _filter_count, new float[filterSize]);
            _filter_weights_errors = new Tensor(_controller, _filter_height, _filter_width, _filter_depth, _filter_count, new float[filterSize]);

            int biasSize = _output_height * _output_width * _output_depth;
            float[] filter_bias = RandomGenerator.GetFloatDistribution(biasSize, 0f, weight_range);
            _filter_bias = new Tensor(_controller, _output_height, _output_width, _output_depth, filter_bias);
            _filter_bias_errors = new Tensor(_controller, _output_height, _output_width, _output_depth, new float[biasSize]);
            _momentum_filter_bias = new Tensor(_controller, _output_height, _output_width, _output_depth, new float[biasSize]);

            _activation = layerActivation;
        }

        public Tensor Forward(Tensor input)
        {
            _inputs = input;
            _controller.ConvolutionForward(_outputs, _inputs, _filter_weights, _filter_bias, _activation);
            return _outputs;
        }

        public Tensor Backward(Tensor errors, float learningRate, bool calculateInputErrors = true)
        {
            //can reuse derive and multiply kernel to get output errors
            _controller.ConvolutionOutputError(_output_errors, _outputs, errors, _activation);
            //need a pad2d
            //need a filter invert
            //need an input error convolution

            //need weighterrorcorrelation
            //momentum update appear the same
            throw new NotImplementedException();
        }


    }
}
