using System;
using DataLabCore;

namespace DataLabConsole
{
    class Program
    {
        static void Main(string[] args)
        {
            int batch_size = 50;
            float learning_rate = 0.01f;
            //TensorController tc = new TensorController(ControllerType.CUDA);
            //ModelTrainer builder = new ModelTrainer();

            //var ds = new DataSource_CIFAR10(tc, batch_size);
            ////var ds = new DataSource_MNIST(tc, batch_size);

            //var conv1 = new ConvolutionLayerTrainer(tc, ds.OutHeight, ds.OutWidth, ds.OutDepth, 3, 3, 32, batch_size, ActivationType.ReLU, PaddingType.Same);
            //var conv2 = new ConvolutionLayerTrainer(tc, conv1.OutHeight, conv1.OutWidth, conv1.OutDepth, 3, 3, 32, batch_size, ActivationType.ReLU, PaddingType.Same);
            //var pool1 = new MaxPoolLayerTrainer(tc, conv2.OutHeight, conv2.OutWidth, conv2.OutDepth, batch_size);

            //var conv3 = new ConvolutionLayerTrainer(tc, pool1.OutHeight, pool1.OutWidth, pool1.OutDepth, 3, 3, 64, batch_size, ActivationType.ReLU, PaddingType.Same);
            //var conv4 = new ConvolutionLayerTrainer(tc, conv3.OutHeight, conv3.OutWidth, conv3.OutDepth, 3, 3, 64, batch_size, ActivationType.ReLU, PaddingType.Same);
            //var pool2 = new MaxPoolLayerTrainer(tc, conv4.OutHeight, conv4.OutWidth, conv4.OutDepth, batch_size);

            //var conv5 = new ConvolutionLayerTrainer(tc, pool2.OutHeight, pool2.OutWidth, pool2.OutDepth, 3, 3, 128, batch_size, ActivationType.ReLU, PaddingType.Same);
            //var conv6 = new ConvolutionLayerTrainer(tc, conv5.OutHeight, conv5.OutWidth, conv5.OutDepth, 3, 3, 128, batch_size, ActivationType.ReLU, PaddingType.Same);
            //var pool3 = new MaxPoolLayerTrainer(tc, conv6.OutHeight, conv6.OutWidth, conv6.OutDepth, batch_size);

            //var flat = new FlattenLayerTrainer(tc, pool3.OutHeight, pool3.OutWidth, pool3.OutDepth, batch_size);
            //var dense = new DenseLayerTrainer(tc, flat.OutSize, 256, batch_size, ActivationType.ReLU);
            //var softmax = new DenseLayerTrainer(tc, dense.OutSize, 10, batch_size, ActivationType.Softmax);
            //var loss = new LossLayer(tc, softmax.OutSize, batch_size, ds.Samples, LossFunction.Multiclass);

            //builder.AddLayer(conv1);
            //builder.AddLayer(conv2);
            //builder.AddLayer(pool1);

            //builder.AddLayer(conv3);
            //builder.AddLayer(conv4);
            //builder.AddLayer(pool2);

            //builder.AddLayer(conv5);
            //builder.AddLayer(conv6);
            //builder.AddLayer(pool3);

            //builder.AddLayer(flat);
            //builder.AddLayer(dense);
            //builder.AddLayer(softmax);
            //builder.AddLossLayer(loss);
            //builder.FitModel(ds, 10000, 0.01f, learning_rate);

            Console.WriteLine("Done");
            Console.ReadLine();
            
        }
    }
}
