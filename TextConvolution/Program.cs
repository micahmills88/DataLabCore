using System;
using DataLabCore;

namespace TextConvolution
{
    class Program
    {
        static void Main(string[] args)
        {
            int batch_size = 50;
            float learning_rate = 0.01f;
            TensorController tc = new TensorController(ControllerType.CUDA);
            ModelBuilder builder = new ModelBuilder();

            var ds = new DataSource_TEXT(tc, 100, batch_size, 50000);

            var conv1 = new ConvolutionLayerT(tc, ds.OutHeight, ds.OutWidth, ds.OutDepth, 5, ds.OutWidth, 512, batch_size, ActivationType.ReLU, PaddingType.None);
            var conv2 = new ConvolutionLayerT(tc, conv1.OutHeight, conv1.OutWidth, conv1.OutDepth, 5, conv1.OutWidth, 512, batch_size, ActivationType.ReLU, PaddingType.None);

            var flat = new FlattenLayerT(tc, conv2.OutHeight, conv2.OutWidth, conv2.OutDepth, batch_size);
            var softmax = new DenseLayerT(tc, flat.OutSize, 64, batch_size, ActivationType.Softmax);
            var loss = new LossLayer(tc, softmax.OutSize, batch_size, ds.Samples, LossFunction.Multiclass);

            builder.AddLayer(conv1);
            builder.AddLayer(conv2);
            builder.AddLayer(flat);
            builder.AddLayer(softmax);
            builder.AddLossLayer(loss);
            builder.FitModel(ds, 10000, 0.01f, learning_rate);            



            Console.WriteLine("Done");
            Console.ReadLine();
        }
    }
}
