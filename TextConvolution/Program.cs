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
                string uri = @"mongodb://10.0.0.21:27017";
                string modelName = "TCN_Save_" + Guid.NewGuid().ToString("N");
                var datasource = new DataSource_TEXT(64, 50000);
                //var datasource = new DataSource_MNIST();

                //var trainer = ModelBuilder.CreateNewModel(uri, modelName, batchSize, datasource, ControllerType.CUDA);
                //trainer.AddConvolutionLayer(3, 3, 32);
                //trainer.AddConvolutionLayer(3, 3, 32);
                //trainer.AddFlattenLayer();
                //trainer.AddDenselayer(256, ActivationType.ReLU);
                //trainer.AddDenselayer(datasource.Classes, ActivationType.Softmax);

                string modelid = "5e8e26f13bbd640f34a195c3";
                var trainer = ModelBuilder.LoadFromDatabase(uri, modelid, batchSize, datasource, ControllerType.CUDA);

                trainer.TrainModel(0.001f, LossFunction.Multiclass, 5, 0.01f);
                trainer.SaveModelToDatabase();

                Console.WriteLine("Done");
                Console.ReadLine();
                datasource = null;
                trainer = null;

            }

            return;
        }
    }
}
