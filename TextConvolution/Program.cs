using System;
using DataLabCore;

namespace TextConvolution
{
    class Program
    {
        static void Main(string[] args)
        {

            var datasource = new DataSource_TEXT(64, 50000);
            var trainer = new ModelBuilder(datasource, ControllerType.CUDA);

            trainer.AddConvolutionLayer(3, 3, 32);
            trainer.AddConvolutionLayer(3, 3, 32);
            trainer.AddFlattenLayer();
            trainer.AddDenselayer(256, ActivationType.ReLU);
            trainer.AddDenselayer(datasource.Classes, ActivationType.Softmax);

            trainer.Initialize(50);
            trainer.TrainModel(1f, LossFunction.Multiclass, 1000, 0.01f);

            Console.WriteLine("Done");
            Console.ReadLine();
        }
    }
}
