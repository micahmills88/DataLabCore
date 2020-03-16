using System;
using DataLabCore;

namespace DataLabConsole
{
    class Program
    {
        static void Main(string[] args)
        {
            int batch_size = 100;
            float learning_rate = 1f;
            TensorController tc = new TensorController();
            ModelBuilder builder = new ModelBuilder();

            var c1 = new ConvolutionLayer(tc, 32, 32, 3, 10, 10, 64, batch_size, ActivationType.ReLU);
            //var c2 = new ConvolutionLayer(tc, c1.OutHeight, c1.OutWidth, c1.OutDepth, 3, 3, 32, batch_size, ActivationType.ReLU);
            //var c3 = new ConvolutionLayer(tc, c2.OutHeight, c2.OutWidth, c2.OutDepth, 3, 3, 64, batch_size, ActivationType.ReLU);
            //var c4 = new ConvolutionLayer(tc, c3.OutHeight, c3.OutWidth, c3.OutDepth, 3, 3, 64, batch_size, ActivationType.ReLU);
            //var fl = new FlattenLayer(c4.OutHeight, c4.OutWidth, c4.OutDepth, batch_size);
            var fl = new FlattenLayer(c1.OutHeight, c1.OutWidth, c1.OutDepth, batch_size);

            builder.AddLayer(c1);
            //builder.AddLayer(c2);
            //builder.AddLayer(c3);
            //builder.AddLayer(c4);
            builder.AddLayer(fl);
            builder.AddLayer(new DenseLayer(tc, fl.OutSize, 2048, batch_size, ActivationType.Sigmoid));
            builder.AddLayer(new DenseLayer(tc, 2048, 10, batch_size, ActivationType.Softmax));
            builder.AddLayer(new LossLayer(tc, 10, batch_size, 10000, LossFunction.Multiclass));
            builder.FitModel(new DataSource_CIFAR10(tc, batch_size), 100, learning_rate);

            Console.WriteLine("Done");
            Console.ReadLine();
            
        }
    }
}
