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
                string uri = @"mongodb://localhost:27017";
                string modelName = "TCN_Test_1234";
                var datasource = new DataSource_TEXT(10, 5000);
                //var datasource = new DataSource_MNIST();

                var trainer = ModelBuilder.CreateNewModel(uri, modelName, batchSize, datasource, ControllerType.CUDA);
                trainer.AddConvolutionLayer(5, 5, 256, ActivationType.ReLU, PaddingType.Same);
                trainer.AddConvolutionLayer(5, 5, 256, ActivationType.ReLU, PaddingType.Same);
                trainer.AddConvolutionLayer(5, 5, 256, ActivationType.ReLU, PaddingType.Same);
                trainer.AddConvolutionLayer(5, 5, 256, ActivationType.ReLU, PaddingType.Same);
                trainer.AddConvolutionLayer(5, 5, 256, ActivationType.ReLU, PaddingType.Same);
                trainer.AddFlattenLayer();
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
