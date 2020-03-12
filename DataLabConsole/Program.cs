using System;
using DataLabCore;

namespace DataLabConsole
{
    class Program
    {
        static void Main(string[] args)
        {
            ModelBuilder builder = new ModelBuilder();
            builder.AddLayer(new DenseLayer(784, 256, 10, ActivationType.Sigmoid));
            builder.AddLayer(new DenseLayer(256, 10, 10, ActivationType.Softmax));
            builder.AddLayer(new LossLayer());
            builder.FitModel(/*datasource*/);
        }
    }
}
