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

            int batch_size = 16;
            DataSource ds = new DataSource(batch_size);
            IModelLayer layer1 = new DenseLayer(784, 256, ActivationType.Sigmoid, batch_size);
            IModelLayer layer2 = new DenseLayer(256, 10, ActivationType.Softmax, batch_size);
            LossCalculator lossCalc = new LossCalculator(LossFunction.Multiclass, 10, batch_size);

            ModelBuilder builder = new ModelBuilder();
            builder.AddLayer(layer1);
            builder.AddLayer(layer2);
            builder.FitModel(ds, lossCalc);
        }
    }
}
