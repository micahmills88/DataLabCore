using System;
using System.Collections.Generic;
using System.Text;

using ILGPU;
using ILGPU.Runtime;

namespace DataLabCore
{
    public class ModelLayer
    {
        private int _inputs;
        private int _outputs;

        private int _weight_count;
        private int _bias_count;

        private LayerType _layer_type;
        private ActivationType _activation_type;

        private int _buffer_size;
        private MemoryBuffer<float> _buffer;
        private ArrayView<float> _weights;
        private ArrayView<float> _bias;
        private ArrayView<float> _momentum_weights;
        private ArrayView<float> _momentum_bias;

        public ModelLayer(Accelerator _accelerator, LayerType layerType, int inputs, int outputs, ActivationType activationType)
        {
            _inputs = inputs;
            _outputs = outputs;
            _weight_count = inputs * outputs;
            _bias_count = outputs;
            _layer_type = layerType;
            _activation_type = activationType;
            _buffer_size = (_weight_count * 2) + (_bias_count * 2);

            _buffer = _accelerator.Allocate<float>(_buffer_size);
            _weights = _buffer.GetSubView(0, _weight_count);
            _momentum_weights = _buffer.GetSubView(_weight_count, _weight_count);
            _bias = _buffer.GetSubView(_weight_count * 2, _bias_count);
            _momentum_bias = _buffer.GetSubView(_weight_count * 2 + _bias_count, _bias_count);
        }
    }
}
