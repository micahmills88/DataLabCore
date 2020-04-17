using System;
using System.Collections.Generic;
using System.Text;

namespace DataLabCore
{
    public class DenseLayer : IProductionLayer
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

        Tensor _inputs;
        Tensor _outputs;

        public DenseLayer(TensorController controller, int inputs, int batchSize, LayerConfig layerConfig)
        {
            _controller = controller;
            _config = layerConfig;

            _input_count = inputs;
            _output_count = layerConfig.WeightColumns;
            _batch_size = batchSize;
            _activation = layerConfig.activationType;

            _weights = new Tensor(_controller, _input_count, _output_count, _config.Weights);
            _bias = new Tensor(_controller, _config.Bias);

            var output_count = _batch_size * _output_count;
            _outputs = new Tensor(_controller, _batch_size, _output_count, new float[output_count]);
        }

        public Tensor Forward(Tensor input)
        {
            _inputs = input; //inputs are always batch_size * num_inputs
            _controller.DenseForward(_outputs, _inputs, _weights, _bias, _activation);
            return _outputs;
        }
    }
}
