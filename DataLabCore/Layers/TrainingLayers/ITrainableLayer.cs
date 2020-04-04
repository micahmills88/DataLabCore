using System;
using System.Collections.Generic;
using System.Text;

namespace DataLabCore
{
    public interface ITrainableLayer : IProductionLayer
    {
        Tensor Backward(Tensor error, float learningRate, bool calculateInputErrors);
        LayerDescription ExportLayerDescription();
    }
}
