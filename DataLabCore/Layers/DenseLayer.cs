using System;
using System.Collections.Generic;
using System.Text;
using DataLabCore.Utils;

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

        Tensor _momentum_weight;
        Tensor _momentum_bias;

        Tensor _inputs;
        Tensor _outputs;

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
            _momentum_weight = new Tensor(_input_count, _output_count, new float[weight_count]);

            _bias = new Tensor(RandomGenerator.GetFloatDistribution(_output_count, 0f, weight_range));
            _momentum_bias = new Tensor(new float[_output_count]);

            var output_count = _batch_size * _output_count;
            _outputs = new Tensor(_batch_size, _output_count, new float[output_count]);
        }
    }
}
