using System;
using System.Collections.Generic;
using System.Text;

namespace DataLabCore
{

    public class NetworkModel
    {
        List<IProductionLayer> layers = new List<IProductionLayer>();

        public NetworkModel()
        {

        }

        public void AddLayer(IProductionLayer layer)
        {
            layers.Add(layer);
        }

        public Tensor Process(Tensor sample)
        {
            Tensor result = layers[0].Forward(sample);
            for (int i = 1; i < layers.Count; i++)
            {
                result = layers[i].Forward(result);
            }
            return result;
        }
    }
}
