using System;
using DataLabCore;

namespace TextConvolution
{
    class Program
    {
        static void Main(string[] args)
        {
            {

                int batchSize = 50;
                string uri = @"mongodb://10.0.0.20:27017";
                string modelName = "MNIST_EXAMPLE";
                //var datasource = new DataSource_TEXT(10, 5000);
                //var datasource = new DataSource_MNIST();
                var datasource = new DataSource_CIFAR10();

                var trainer = ModelBuilder.CreateNewModel(uri, modelName, batchSize, datasource, ControllerType.CUDA);
                trainer.AddConvolutionLayer(3, 3, 32, ActivationType.ReLU, PaddingType.Same);
                trainer.AddConvolutionLayer(3, 3, 32, ActivationType.ReLU, PaddingType.Same);
                trainer.AddMaxPoolLayer();
                trainer.AddConvolutionLayer(3, 3, 64, ActivationType.ReLU, PaddingType.Same);
                trainer.AddConvolutionLayer(3, 3, 64, ActivationType.ReLU, PaddingType.Same);
                trainer.AddMaxPoolLayer();
                trainer.AddFlattenLayer();
                trainer.AddDenselayer(256, ActivationType.Sigmoid);
                trainer.AddDenselayer(datasource.Classes, ActivationType.Softmax);

                //string modelid = "5e8e26f13bbd640f34a195c3";
                //var trainer = ModelBuilder.LoadFromDatabase(uri, modelid, batchSize, datasource, ControllerType.CUDA);

                trainer.TrainModel(0.01f, LossFunction.Multiclass, 100, 0.01f);
                //trainer.SaveModelToDatabase();

                Console.WriteLine("Done");
                Console.ReadLine();
                datasource = null;
                trainer = null;

            }

            return;
        }
    }
}
