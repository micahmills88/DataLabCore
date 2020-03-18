using System;
using DataLabCore;

namespace DataLabConsole
{
    class Program
    {
        static void Main(string[] args)
        {
            int batch_size = 30;
            float learning_rate = 0.1f;
            TensorController tc = new TensorController(ControllerType.CUDA);
            ModelBuilder builder = new ModelBuilder();

            var ds = new DataSource_CIFAR10(tc, batch_size);

            var c1 = new ConvolutionLayer(tc, ds.OutHeight, ds.OutWidth, ds.OutDepth, 3, 3, 16, batch_size, ActivationType.ReLU);
            var c2 = new ConvolutionLayer(tc, c1.OutHeight, c1.OutWidth, c1.OutDepth, 3, 3, 16, batch_size, ActivationType.ReLU);
            var mp1 = new MaxPoolLayer(tc, c2.OutHeight, c2.OutWidth, c2.OutDepth, batch_size);

            var c3 = new ConvolutionLayer(tc, mp1.OutHeight, mp1.OutWidth, mp1.OutDepth, 3, 3, 32, batch_size, ActivationType.ReLU);
            var c4 = new ConvolutionLayer(tc, c3.OutHeight, c3.OutWidth, c3.OutDepth, 3, 3, 32, batch_size, ActivationType.ReLU);
            var mp2 = new MaxPoolLayer(tc, c4.OutHeight, c4.OutWidth, c4.OutDepth, batch_size);

            var fl = new FlattenLayer(mp2.OutHeight, mp2.OutWidth, mp2.OutDepth, batch_size);
            var d1 = new DenseLayer(tc, fl.OutSize, 512, batch_size, ActivationType.ReLU);
            var d2 = new DenseLayer(tc, d1.OutCount, 10, batch_size, ActivationType.Softmax);
            var l1 = new LossLayer(tc, d2.OutCount, batch_size, ds.Samples, LossFunction.Multiclass);

            builder.AddLayer(c1);
            builder.AddLayer(c2);
            builder.AddLayer(mp1);
            builder.AddLayer(c3);
            builder.AddLayer(c4);
            builder.AddLayer(mp2);
            builder.AddLayer(fl);
            builder.AddLayer(d1);
            builder.AddLayer(d2);
            builder.AddLayer(l1);

            builder.FitModel(ds, 100, learning_rate);

            Console.WriteLine("Done");
            Console.ReadLine();
            
        }
    }
}
