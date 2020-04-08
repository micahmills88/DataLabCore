using System;
using System.Collections.Generic;
using System.Text;

namespace DataLabCore
{
    public class DenseLayerBuilder : ITrainableLayer
    {
        TensorController _controller;
        LayerConfig _config;

        int _input_count;
        int _output_count;
        int _batch_size;
        ActivationType _activation;

        public int OutputHeight { get => _batch_size; }
        public int OutputWidth { get => _output_count; }
        public int OutputDepth { get => 1; }
        public int OutputSize { get => _output_count; }

        Tensor _weights;
        Tensor _bias;

        Tensor _weights_errors;
        Tensor _bias_errors;

        Tensor _momentum_weights;
        Tensor _momentum_bias;

        Tensor _inputs;
        Tensor _outputs;

        Tensor _input_errors;
        Tensor _output_errors;

        public DenseLayerBuilder(TensorController controller, int inputs, int batchSize, LayerConfig layerConfig, bool genWeights)
        {
            _controller = controller;
            _config = layerConfig;

            _input_count = inputs;
            _output_count = layerConfig.WeightColumns;
            _batch_size = batchSize;
            _activation = layerConfig.activationType;

            //default is glorot (sigmoid)
            //bool isRelu = _activation == ActivationType.ReLU;

            var weight_count = _input_count * _output_count;
            //if(isRelu)
            //{
            //    float weight_range = (float)Math.Sqrt(2.0d / (float)_input_count);
            //    _config.Weights = RandomGenerator.GetFloatNormalDistribution(weight_count, 0f, weight_range);
            //    _config.Bias = new float[_output_count];
            //}
            //else
            //{
            //    float weight_range = (float)Math.Sqrt(2.0d / (float)_input_count);
            //    _config.Weights = RandomGenerator.GetFloatNormalDistribution(weight_count, 0f, weight_range);
            //    _config.Bias = RandomGenerator.GetFloatNormalDistribution(_output_count, 0f, weight_range);
            //}

            if (genWeights || _config.Weights == null)
            {
                float weight_range = (float)Math.Sqrt(2.0d / (float)_input_count);
                _config.Weights = RandomGenerator.GetFloatNormalDistribution(weight_count, 0f, weight_range);
                _config.Bias = new float[_output_count];
            }


            _weights = new Tensor(_controller, _input_count, _output_count, _config.Weights);
            _momentum_weights = new Tensor(_controller, _input_count, _output_count, new float[weight_count]);
            _weights_errors = new Tensor(_controller, _input_count, _output_count, new float[weight_count]);

            _bias = new Tensor(_controller, _config.Bias);
            _momentum_bias = new Tensor(_controller, new float[_output_count]);
            _bias_errors = new Tensor(_controller, new float[_output_count]);

            var output_count = _batch_size * _output_count;
            _outputs = new Tensor(_controller, _batch_size, _output_count, new float[output_count]);
            _output_errors = new Tensor(_controller, _batch_size, _output_count, new float[output_count]);

            var error_count = _input_count * _batch_size;
            _input_errors = new Tensor(_controller, _input_count, _batch_size, new float[error_count]);
        }

        public Tensor Forward(Tensor input)
        {
            _inputs = input; //inputs are always batch_size * num_inputs
            _controller.DenseForward(_outputs, _inputs, _weights, _bias, _activation);
            return _outputs;
        }

        public Tensor Backward(Tensor errors, float learningRate, bool calculateInputErrors = true)
        {
            //output_errors start out as batch_size * num_output
            _controller.DenseOutputError(_output_errors, _outputs, errors, _activation);
            //dense output error transposes the out_errors to num_output * batch_size
            if(calculateInputErrors)
            {
                _controller.DenseInputError(_input_errors, _weights, _output_errors);
            }
            float batchMultiple = (1.0f / (float)_batch_size);
            _controller.DenseLayerWeightUpdate(_weights, _weights_errors, _momentum_weights, _inputs, _output_errors, batchMultiple, learningRate);
            _controller.DenseLayerBiasUpdate(_bias, _bias_errors, _momentum_bias, _output_errors, batchMultiple, learningRate);
            //we need to reset _output_errors dimensions before the next loop
            _output_errors.Transpose2DValues();
            return _input_errors;
        }

        public LayerConfig ExportLayer()
        {
            _config.layerType = LayerType.Dense;
            _config.HasWeights = true;
            _config.WeightRows = _weights.Rows;
            _config.WeightColumns = _weights.Columns;
            _config.WeightLayers = _weights.Layers;
            _config.WeightCubes = _weights.Cubes;
            _config.HasBias = true;
            _config.BiasRows = _bias.Rows;
            _config.BiasColumns = _bias.Columns;
            _config.BiasLayers = _bias.Layers;
            _config.BiasCubes = _bias.Cubes;
            _config.activationType = _activation;
            _config.paddingType = PaddingType.None;

            _weights.SynchronizeToLocal();
            _bias.SynchronizeToLocal();

            return _config;
        }
    }
}
