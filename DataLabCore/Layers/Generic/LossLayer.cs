﻿using System;
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

    public class LossLayer
    {
        TensorController _controller;

        int _input_size;
        int _batch_size;
        int _epoch_size;

        Tensor _errors;
        Tensor _epoch_error;

        public LossLayer(TensorController controller, int inputSize, int batchSize, int epochSize)
        {
            _controller = controller;
            _input_size = inputSize;
            _batch_size = batchSize;
            _epoch_size = epochSize;

            var error_size = _batch_size * _input_size;
            _errors = new Tensor(_controller, _input_size, _batch_size, new float[error_size]);
            _epoch_error = new Tensor(_controller, _batch_size, _input_size, new float[error_size]);
        }

        public Tensor CalculateLoss(Tensor predictions, Tensor labels, LossFunction lossFunction)
        {
            _controller.CalculateLoss(_epoch_error, _errors, predictions, labels, lossFunction);
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
