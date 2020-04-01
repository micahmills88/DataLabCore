using System;
using System.Collections.Generic;
using System.Text;

namespace DataLabCore
{
    public interface ITrainableLayer
    {
        Tensor Forward(Tensor data);
        Tensor Backward(Tensor error, float learningRate, bool calculateInputErrors);
    }

    public enum ActivationType
    {
        Sigmoid,
        Softmax,
        ReLU
    }

    public enum PaddingType
    {
        None,
        Same
    }
}
