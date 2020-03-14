using System;
using DataLabCore;

namespace DataLabConsole
{
    class Program
    {
        static void Main(string[] args)
        {
            int batch_size = 10;
            float learning_rate = 1f;
            TensorController tc = new TensorController();
            ModelBuilder builder = new ModelBuilder();
            builder.AddLayer(new DenseLayer(tc, 3072, 8192, batch_size, ActivationType.Sigmoid));
            builder.AddLayer(new DenseLayer(tc, 8192, 4096, batch_size, ActivationType.Sigmoid));
            builder.AddLayer(new DenseLayer(tc, 4096, 2048, batch_size, ActivationType.Sigmoid));
            builder.AddLayer(new DenseLayer(tc, 2048, 10, batch_size, ActivationType.Softmax));
            builder.AddLayer(new LossLayer(tc, 10, batch_size, 10000, LossFunction.Multiclass));
            builder.FitModel(new DataSource_CIFAR10(tc, batch_size), 20, learning_rate);

            Console.WriteLine("Done");
            Console.ReadLine();
            
        }
    }
}
