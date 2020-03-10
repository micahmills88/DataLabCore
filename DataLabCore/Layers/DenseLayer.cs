using System;
using System.Collections.Generic;
using System.Text;
using DataLabCore.Utils;
using ILGPU;
using ILGPU.Algorithms;
using ILGPU.Runtime;

namespace DataLabCore
{
    public partial class DenseLayer : IModelLayer
    {
        int _num_inputs;
        int _num_outputs;
        int _batch_size;
        int _weight_count;
        int _bias_count;

        int _input_size;
        int _output_size;

        ActivationType _activation_type;

        int _buffer_size;
        MemoryBuffer<float> _buffer;
        ArrayView<float> _weights;
        ArrayView<float> _bias;
        ArrayView<float> _momentum_weights;
        ArrayView<float> _momentum_bias;
        ArrayView<float> _outputs;
        ArrayView<float> _layer_error;
        ArrayView<float> _return_error;

        public DenseLayer(int inputs, int outputs, ActivationType activationType, int batchSize = 10, float learningRate = 1.0f)
        {
            _num_inputs = inputs;
            _num_outputs = outputs;
            _batch_size = batchSize;
            _weight_count = inputs * outputs;
            _bias_count = outputs;
            _activation_type = activationType;

            _input_size = _num_inputs * _batch_size;
            _output_size = _num_outputs * _batch_size;
            _buffer_size = (_weight_count * 2) + (_bias_count * 2) + (_output_size * 2) + _input_size;
        }

        public void Initialize(Accelerator accelerator, RandomGenerator _random)
        {
            _buffer = accelerator.Allocate<float>(_buffer_size);

            int offset = 0;
            _weights = _buffer.GetSubView(offset, _weight_count);
            offset += _weight_count;

            _momentum_weights = _buffer.GetSubView(offset, _weight_count);
            offset += _weight_count;

            _bias = _buffer.GetSubView(offset, _bias_count);
            offset += _bias_count;

            _momentum_bias = _buffer.GetSubView(offset, _bias_count);
            offset += _bias_count;

            _outputs = _buffer.GetSubView(offset, _output_size);
            offset += _output_size;

            _layer_error = _buffer.GetSubView(offset, _output_size);
            offset += _output_size;

            _return_error = _buffer.GetSubView(offset, _input_size);
            offset += _input_size;

            float weight_range = (float)Math.Sqrt(1.0d / (float)_num_inputs);
            var initial_weights = _random.GetFloatDistribution(_weight_count, 0f, weight_range);
            var initial_bias = _random.GetFloatDistribution(_bias_count, 0f, weight_range);

            _buffer.MemSetToZero();
            _buffer.CopyFrom(initial_weights, 0, 0, _weights.Extent);
            _buffer.CopyFrom(initial_bias, 0, _weight_count * 2, _bias.Extent);

            k_dense_forward = accelerator.LoadAutoGroupedStreamKernel<ILGPU.Index, ArrayView<float>, ArrayView<float>, ArrayView<float>, ArrayView<float>, int, int>(DenseForward);
            k_activate_sigmoid = accelerator.LoadAutoGroupedStreamKernel<ILGPU.Index, ArrayView<float>>(ActivateSigmoid);
        }

        public ArrayView<float> Forward(Accelerator _accelerator, ArrayView<float> inputs)
        {
            //process forward kernel
            k_dense_forward(_outputs.Length, inputs, _weights, _bias, _outputs, _num_inputs, _num_outputs);
            _accelerator.Synchronize();

            //process activation
            if (_activation_type == ActivationType.Sigmoid)
            {
                k_activate_sigmoid(_outputs.Length, _outputs);
            }
            _accelerator.Synchronize();
            return _outputs;
        }

        public ArrayView<float> Backward(Accelerator _accelerator, ArrayView<float> errors, bool calculateReturnErrors = true)
        {
            //the errors coming in should be transposed already
            //calculate the derivative of our last outputs and transpose those
            //pointwise multiply the derivatives by the errors
            //the bias errors are the row sums of those new errors
            //the weight errors are the inputs transposed multiplied by the errors transposed
            //weight and bias momentum updates can be combined with their calculations


            //need to allocate space for output errors the size of the output (_derivatives)
            //need to allocate another memory chunk the size of inputs for return errors (_return_errors

            return errors;
        }

        

        //kernel delegates
        Action<ILGPU.Index, ArrayView<float>, ArrayView<float>, ArrayView<float>, ArrayView<float>, int, int> k_dense_forward;
        Action<ILGPU.Index, ArrayView<float>> k_activate_sigmoid;
        Action<ILGPU.Index, ArrayView<float>> k_derive_sigmoid;
    }
}
