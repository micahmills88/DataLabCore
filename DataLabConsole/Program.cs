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
            builder.AddLayer(new DenseLayer(tc, 784, 512, batch_size, ActivationType.Sigmoid));
            builder.AddLayer(new DenseLayer(tc, 512, 256, batch_size, ActivationType.Sigmoid));
            builder.AddLayer(new DenseLayer(tc, 256, 10, batch_size, ActivationType.Softmax));
            builder.AddLayer(new LossLayer(tc, 10, batch_size, 60000, LossFunction.Multiclass));
            builder.FitModel(new DataSource(tc, batch_size), 10, learning_rate);

            Console.WriteLine("Done");
            Console.ReadLine();
            
        }
    }
}
