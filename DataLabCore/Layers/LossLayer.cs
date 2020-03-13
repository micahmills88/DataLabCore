using System;
using System.Collections.Generic;
using System.Text;
using System.Linq;

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
        TensorController _controller;

        int _input_size;
        int _batch_size;
        int _epoch_size;
        LossFunction _loss_function;

        Tensor _inputs;
        Tensor _errors;
        Tensor _epoch_error;

        public LossLayer(TensorController controller, int inputSize, int batchSize, int epochSize, LossFunction lossFunction)
        {
            _controller = controller;
            _input_size = inputSize;
            _batch_size = batchSize;
            _epoch_size = epochSize;
            _loss_function = lossFunction;

            var error_size = _batch_size * _input_size;
            _errors = new Tensor(_controller, _input_size, _batch_size, new float[error_size]);
            _epoch_error = new Tensor(_controller, _batch_size, _input_size, new float[error_size]);
        }

        public Tensor Forward(Tensor predictions)
        {
            _inputs = predictions;
            return predictions;
        }

        public Tensor Backward(Tensor labels, float learningRate, bool calculateInputErrors = true)
        {
            _controller.CalculateLoss(_epoch_error, _errors, _inputs, labels, _loss_function);
            return _errors;
        }

        public float GetEpochLoss()
        {
            _epoch_error.SynchronizeToLocal();
            var loss = _epoch_error.Data.Sum() / (float)_epoch_size;
            _epoch_error.ZeroMemory();
            return loss;
        }
    }
}
