using System;
using System.Collections.Generic;
using System.Linq;
using DataLabCore;

namespace GeoConverter
{
    class Program
    {
        static void Main(string[] args)
        {
            // int batchSize = 10;
            // string uri = @"mongodb://10.0.0.21:27017";
            // string modelName = "GEO_CONVERTER";
            // //var datasource = new DataSource_TEXT(10, 5000);
            // //var datasource = new DataSource_MNIST();
            // var datasource = new DataSource_GEO();

            // var trainer = ModelBuilder.CreateNewModel(uri, modelName, batchSize, datasource, ControllerType.CUDA);
            // trainer.AddDenselayer(10, ActivationType.Sigmoid);
            // trainer.AddDenselayer(100, ActivationType.Sigmoid);
            // trainer.AddDenselayer(1000, ActivationType.Sigmoid);
            // trainer.AddDenselayer(100, ActivationType.Sigmoid);
            // trainer.AddDenselayer(10, ActivationType.Sigmoid);
            // trainer.AddDenselayer(datasource.Classes, ActivationType.Sigmoid);

            // //string modelid = "5e8e26f13bbd640f34a195c3";
            // //var trainer = ModelBuilder.LoadFromDatabase(uri, modelid, batchSize, datasource, ControllerType.CUDA);

            // trainer.TrainModel(0.1f, LossFunction.MeanSquared, 1000, 0.01f);
            // //trainer.SaveModelToDatabase();

            // Console.WriteLine("Done");
            // Console.ReadLine();
            // datasource = null;
            // trainer = null;
        }
    }
}
