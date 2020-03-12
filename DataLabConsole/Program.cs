using System;
using DataLabCore;

namespace DataLabConsole
{
    class Program
    {
        static void Main(string[] args)
        {
            int batch_size = 16;
            DataSource ds = new DataSource(batch_size);

            ModelBuilder builder = new ModelBuilder();
            builder.AddLayer(new DenseLayer(784, 256, batch_size, ActivationType.Sigmoid));
            builder.AddLayer(new DenseLayer(256, 10, batch_size, ActivationType.Softmax));
            builder.AddLayer(new LossLayer(10, batch_size, LossFunction.Multiclass));
            builder.FitModel(ds, 10);

            Console.WriteLine("Done");
            Console.ReadLine();
        }
    }
}
