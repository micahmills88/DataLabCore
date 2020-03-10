using DataLabCore.Utils;
using ILGPU;
using ILGPU.Runtime;
using System;
using System.Collections.Generic;
using System.Text;

namespace DataLabCore
{
    public class LossCalculator
    {
        int _batch_size;
        LossFunction _loss_function;

        int _buffer_size;
        int _inputs;

        MemoryBuffer<float> _buffer;
        ArrayView<float> _outputs;

        public LossCalculator(LossFunction lossFunction, int inputs, int batchSize)
        {
            _loss_function = lossFunction;
            _inputs = inputs;
            _batch_size = batchSize;

            _buffer_size = _inputs * _batch_size;
        }        

        public void Initialize(Accelerator accelerator)
        {
            _buffer = accelerator.Allocate<float>(_buffer_size);
            _buffer.MemSetToZero();
            _outputs = _buffer.View;
        }

        public ArrayView<float> CalculateLoss(Accelerator _accelerator, ArrayView<float> predictions, ArrayView<float> labels)
        {
            k_calculate_loss(_outputs.Length, predictions, labels, _outputs);
            return _outputs;
        }

        Action<ILGPU.Index, ArrayView<float>, ArrayView<float>, ArrayView<float>> k_calculate_loss;

        private void CalculateLoss(
            ILGPU.Index index,
            ArrayView<float> predictions,
            ArrayView<float> labels,
            ArrayView<float> output
        )
        {
            output[index] = predictions[index] - labels[index];
        }
    }
}
