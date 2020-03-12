using System;
using System.Collections.Generic;
using System.Text;

namespace DataLabCore
{
    public enum LossFunction
    {
        MeanSquared,
        Logistic,
        Multiclass
    }

    public class LossLayer : IModelLayer
    {
        int _input_size;
        int _batch_size;
        LossFunction _loss_function;

        Tensor _inputs;
        Tensor _errors;
        Tensor _epoch_error;

        public LossLayer(int inputSize, int batchSize, LossFunction lossFunction)
        {
            _input_size = inputSize;
            _batch_size = batchSize;
            _loss_function = lossFunction;

            var error_size = _batch_size * _input_size;
            _errors = new Tensor(_input_size, _batch_size, new float[error_size]);
            _epoch_error = new Tensor(_batch_size, _input_size, new float[error_size]);
        }

        public Tensor Forward(Tensor predictions)
        {
            _inputs = predictions;
            return predictions;
        }

        public Tensor Backward(Tensor labels, bool calculateInputErrors = true)
        {
            TensorController.Instance.CalculateLoss(_epoch_error, _errors, _inputs, labels, _loss_function);
            return _errors;
        }
    }
}
