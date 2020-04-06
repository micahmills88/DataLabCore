using System;
using System.Collections.Generic;
using System.Text;

namespace DataLabCore
{
    public class DenseLayerTrainer : ITrainableLayer
    {
        TensorController _controller;
        LayerDescription _description;

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

        public DenseLayerTrainer(TensorController controller, int inputs, int batchSize, LayerDescription layerDescription)
        {
            _controller = controller;
            _description = layerDescription;

            _input_count = inputs;
            _output_count = layerDescription.WeightColumns;
            _batch_size = batchSize;
            _activation = layerDescription.activationType;

            //default is glorot (sigmoid)
            bool isRelu = _activation == ActivationType.ReLU;

            var weight_count = _input_count * _output_count;
            if(isRelu)
            {
                float weight_range = (float)Math.Sqrt(2.0d / (float)_input_count);
                _description.Weights = RandomGenerator.GetFloatNormalDistribution(weight_count, 0f, weight_range);
                _description.Bias = new float[_output_count];
            }
            else
            {
                float weight_range = (float)Math.Sqrt(2.0d / (float)_input_count);
                _description.Weights = RandomGenerator.GetFloatNormalDistribution(weight_count, 0f, weight_range);
                _description.Bias = RandomGenerator.GetFloatNormalDistribution(_output_count, 0f, weight_range);
            }


            _weights = new Tensor(_controller, _input_count, _output_count, _description.Weights);
            _momentum_weights = new Tensor(_controller, _input_count, _output_count, new float[weight_count]);
            _weights_errors = new Tensor(_controller, _input_count, _output_count, new float[weight_count]);

            _bias = new Tensor(_controller, _description.Bias);
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

        public LayerDescription ExportLayerDescription()
        {
            _description.layerType = LayerType.Dense;
            _description.HasWeights = true;
            _description.WeightRows = _weights.Rows;
            _description.WeightColumns = _weights.Columns;
            _description.WeightLayers = _weights.Layers;
            _description.WeightCubes = _weights.Cubes;
            _description.HasBias = true;
            _description.BiasRows = _bias.Rows;
            _description.BiasColumns = _bias.Columns;
            _description.BiasLayers = _bias.Layers;
            _description.BiasCubes = _bias.Cubes;
            _description.activationType = _activation;
            _description.paddingType = PaddingType.None;

            _weights.SynchronizeToLocal();
            _bias.SynchronizeToLocal();

            return _description;
        }
    }
}
