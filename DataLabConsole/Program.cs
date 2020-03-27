﻿using System;
using DataLabCore;

namespace DataLabConsole
{
    class Program
    {
        static void Main(string[] args)
        {
            int batch_size = 50;
            float learning_rate = 0.01f;
            TensorController tc = new TensorController(ControllerType.CUDA);
            ModelBuilder builder = new ModelBuilder();

            var ds = new DataSource_CIFAR10(tc, batch_size);
            //var ds = new DataSource_MNIST(tc, batch_size);

            var conv1 = new ConvolutionLayer(tc, ds.OutHeight, ds.OutWidth, ds.OutDepth, 3, 3, 32, batch_size, ActivationType.ReLU, PaddingType.Same);
            var conv2 = new ConvolutionLayer(tc, conv1.OutHeight, conv1.OutWidth, conv1.OutDepth, 3, 3, 32, batch_size, ActivationType.ReLU, PaddingType.Same);
            var pool1 = new MaxPoolLayer(tc, conv2.OutHeight, conv2.OutWidth, conv2.OutDepth, batch_size);

            var conv3 = new ConvolutionLayer(tc, pool1.OutHeight, pool1.OutWidth, pool1.OutDepth, 3, 3, 64, batch_size, ActivationType.ReLU, PaddingType.Same);
            var conv4 = new ConvolutionLayer(tc, conv3.OutHeight, conv3.OutWidth, conv3.OutDepth, 3, 3, 64, batch_size, ActivationType.ReLU, PaddingType.Same);
            var pool2 = new MaxPoolLayer(tc, conv4.OutHeight, conv4.OutWidth, conv4.OutDepth, batch_size);

            var conv5 = new ConvolutionLayer(tc, pool2.OutHeight, pool2.OutWidth, pool2.OutDepth, 3, 3, 128, batch_size, ActivationType.ReLU, PaddingType.Same);
            var conv6 = new ConvolutionLayer(tc, conv5.OutHeight, conv5.OutWidth, conv5.OutDepth, 3, 3, 128, batch_size, ActivationType.ReLU, PaddingType.Same);
            var pool3 = new MaxPoolLayer(tc, conv6.OutHeight, conv6.OutWidth, conv6.OutDepth, batch_size);

            var flat = new FlattenLayer(tc, pool3.OutHeight, pool3.OutWidth, pool3.OutDepth, batch_size);
            var dense = new DenseLayer(tc, flat.OutSize, 256, batch_size, ActivationType.ReLU);
            var softmax = new DenseLayer(tc, dense.OutSize, 10, batch_size, ActivationType.Softmax);
            var loss = new LossLayer(tc, softmax.OutSize, batch_size, ds.Samples, LossFunction.Multiclass);

            builder.AddLayer(conv1);
            builder.AddLayer(conv2);
            builder.AddLayer(pool1);

            builder.AddLayer(conv3);
            builder.AddLayer(conv4);
            builder.AddLayer(pool2);

            builder.AddLayer(conv5);
            builder.AddLayer(conv6);
            builder.AddLayer(pool3);

            builder.AddLayer(flat);
            builder.AddLayer(dense);
            builder.AddLayer(softmax);
            builder.AddLossLayer(loss);
            builder.FitModel(ds, 10000, 0.01f, learning_rate);

            Console.WriteLine("Done");
            Console.ReadLine();
            
        }
    }
}
