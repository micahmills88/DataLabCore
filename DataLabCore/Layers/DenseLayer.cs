using System;
using System.Collections.Generic;
using System.Text;

namespace DataLabCore
{
    public class DenseLayer : IModelLayer
    {
        int _input_count;
        int _output_count;
        int _batch_size;
        ActivationType _activation;

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

        public DenseLayer(int inputs, int outputs, int batchSize, ActivationType activationType)
        {
            _input_count = inputs;
            _output_count = outputs;
            _batch_size = batchSize;
            _activation = activationType;

            
            float weight_range = (float)Math.Sqrt(1.0d / (float)_input_count);
            var weight_count = _input_count * _output_count;
            var weight_data = RandomGenerator.GetFloatDistribution(weight_count, 0f, weight_range);
            _weights = new Tensor(_input_count, _output_count, weight_data);
            _momentum_weights = new Tensor(_input_count, _output_count, new float[weight_count]);
            _weights_errors = new Tensor(_input_count, _output_count, new float[weight_count]);

            _bias = new Tensor(RandomGenerator.GetFloatDistribution(_output_count, 0f, weight_range));
            _momentum_bias = new Tensor(new float[_output_count]);
            _bias_errors = new Tensor(new float[_output_count]);

            var output_count = _batch_size * _output_count;
            _outputs = new Tensor(_batch_size, _output_count, new float[output_count]);
            _output_errors = new Tensor(_batch_size, _output_count, new float[output_count]);

            var error_count = _input_count * _batch_size;
            _input_errors = new Tensor(_input_count, _batch_size, new float[error_count]);
        }

        public Tensor Forward(Tensor input)
        {
            _inputs = input;
            TensorController.Instance.DenseForward(_outputs, _inputs, _weights, _bias, _activation);
            return _outputs;
        }

        public Tensor Backward(Tensor errors, bool calculateInputErrors = true)
        {
            var controller = TensorController.Instance;
            controller.DenseOutputError(_output_errors, _outputs, errors, _activation);
            if(calculateInputErrors)
            {
                controller.DenseInputError(_input_errors, _weights, _output_errors);
            }
            float batchMultiple = (1.0f / (float)_batch_size);
            float learningRate = 1.0f;
            controller.DenseLayerWeightUpdate(_weights, _weights_errors, _momentum_weights, _inputs, _output_errors, batchMultiple, learningRate);
            controller.DenseLayerBiasUpdate(_bias, _bias_errors, _momentum_bias, _output_errors, batchMultiple, learningRate);
            return _input_errors;
        }
    }
}
