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
            /*
             * Unlike the ModelTrainer, the ModelRunner does not utilize
             * a loss layer, and returns its results directly.
             * 
             * For a situation such as transfer learning or using prebuilt models
             * the output from a model runner can be fed into another model trainer.
             * 
             * TODO: implement IDATASOURCE here, or wrap this class in a ModelDataSource class
             * (A model datasource would be useful in the future for complex designs)
             */
            Tensor result = layers[0].Forward(sample);
            for (int i = 1; i < layers.Count; i++)
            {
                result = layers[i].Forward(result);
            }
            return result;
        }
    }
}
