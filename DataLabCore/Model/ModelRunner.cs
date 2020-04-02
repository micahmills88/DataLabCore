using System;
using System.Collections.Generic;
using System.Text;

namespace DataLabCore
{

    public class ModelRunner
    {
        List<IProductionLayer> layers = new List<IProductionLayer>();

        public ModelRunner()
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
