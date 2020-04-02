using System;
using System.Collections.Generic;
using System.Text;

namespace DataLabCore
{
    public interface IProductionLayer
    {
        int OutputHeight { get; }
        int OutputWidth { get; }
        int OutputDepth { get; }
        int OutputSize { get; }
        Tensor Forward(Tensor data);
    }
}
