using System;
using System.Collections.Generic;
using System.Text;

namespace DataLabCore
{
    public enum LossFunction
    {
        MSE
    }

    public class LossLayer : IModelLayer
    {
        int _input_size;
        int _batch_size;
        LossFunction _loss_function;

        public LossLayer(int inputSize, int batchSize, LossFunction lossFunction)
        {
            _input_size = inputSize;
            _batch_size = batchSize;
            _loss_function = lossFunction;

            //build your own tensors for data storage
        }

        public Tensor Forward(Tensor data)
        {
            //store the reference to the input tensor
            //the return values dont matter here
            return data;
        }

        public Tensor Backward(Tensor error, bool calculateInputErrors)
        {
            //subtract the tensors
            //perform any calculations
            //store globally for epoch errors?
            return error;
        }
    }
}
