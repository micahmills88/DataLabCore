﻿using System;
using System.Collections.Generic;
using System.Text;

namespace DataLabCore
{
    public interface IModelLayer
    {
        Tensor Forward(Tensor data);
        Tensor Backward(Tensor error, bool calculateInputErrors);
    }

    public enum ActivationType
    {
        Sigmoid,
        Softmax
    }
}
