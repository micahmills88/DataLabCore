using System;
using System.Collections.Generic;
using System.IO;

namespace DataLabCore
{
    class Program
    {
        static void Main(string[] args)
        {
            /*
             * what are some future features that should be considered?
             * 1.) transfer learning - might want an entire other model to be used as input to new layers
             * 2.) non-block datasources such as from a camera or graphics output (screencap)
             * 3.) distribution of processing - for evolutionary strategies or reinforcement learning
             */

            ModelBuilder builder = new ModelBuilder();
            builder.AddLayer(LayerType.Dense, 784, 256, ActivationType.Sigmoid);
            builder.AddLayer(LayerType.Dense, 256, 10, ActivationType.Softmax);

            DataSource ds = new DataSource();
            var batchSize = 16;
            builder.FitModel(ds, LossFunction.Multiclass, batchSize);
        }
    }
}
