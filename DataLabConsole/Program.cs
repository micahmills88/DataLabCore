using System;
using DataLabCore;

namespace DataLabConsole
{
    class Program
    {
        static void Main(string[] args)
        {
            int batch_size = 50;
            float learning_rate = 0.001f;
            TensorController tc = new TensorController(ControllerType.CUDA);
            ModelBuilder builder = new ModelBuilder();

            var ds = new DataSource_CIFAR10(tc, batch_size);
            //var ds = new DataSource_MNIST(tc, batch_size);

            var conv1 = new ConvolutionLayer(tc, ds.OutHeight, ds.OutWidth, ds.OutDepth, 3, 3, 32, batch_size, ActivationType.ReLU, PaddingType.None);
            var conv2 = new ConvolutionLayer(tc, conv1.OutHeight, conv1.OutWidth, conv1.OutDepth, 3, 3, 32, batch_size, ActivationType.ReLU, PaddingType.None);
            var pool1 = new MaxPoolLayer(tc, conv2.OutHeight, conv2.OutWidth, conv2.OutDepth, batch_size);

            var flat = new FlattenLayer(tc, pool1.OutHeight, pool1.OutWidth, pool1.OutDepth, batch_size);
            var dense = new DenseLayer(tc, flat.OutSize, 128, batch_size, ActivationType.ReLU);
            var softmax = new DenseLayer(tc, dense.OutSize, 10, batch_size, ActivationType.Softmax);
            var loss = new LossLayer(tc, softmax.OutSize, batch_size, ds.Samples, LossFunction.Multiclass);

            builder.AddLayer(conv1);
            builder.AddLayer(conv2);
            builder.AddLayer(pool1);

            builder.AddLayer(flat);
            builder.AddLayer(dense);
            builder.AddLayer(softmax);
            builder.AddLossLayer(loss);
            builder.FitModel(ds, 10000, learning_rate);

            Console.WriteLine("Done");
            Console.ReadLine();
            
        }
    }
}
