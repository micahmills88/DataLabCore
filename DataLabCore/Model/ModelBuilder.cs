using System;
using System.Collections.Generic;
using System.Text;
using System.Linq;

using ILGPU;
using ILGPU.Runtime;
using ILGPU.Runtime.Cuda;
using System.Diagnostics;

namespace DataLabCore
{
    public class ModelBuilder
    {
        List<IModelLayer> _layers = new List<IModelLayer>();
        Stopwatch stopwatch = new Stopwatch();

        public ModelBuilder()
        {

        }

        public void AddLayer(IModelLayer layer)
        {
            _layers.Add(layer);
        }

        public void FitModel(IDataSource dataSource, int epochs, float learningRate)
        {
            for (int e = 0; e < epochs; e++)
            {
                stopwatch.Restart();
                for (int b = 0; b < dataSource.GetTotalBatches(); b++)
                {
                    var data = dataSource.GetSampleBatch(b);
                    for (int i = 0; i < _layers.Count; i++)
                    {
                        data = _layers[i].Forward(data);
                    }
                    var error = dataSource.GetLabelBatch(b);
                    for (int i = _layers.Count - 1; i >= 0; i--)
                    {
                        error = _layers[i].Backward(error, learningRate, i > 0);
                    }
                }
                
                var losslayer = (LossLayer)(_layers.Last() as LossLayer);
                var loss = losslayer.GetEpochLoss();
                stopwatch.Stop();
                Console.WriteLine("Epoch: {0} Time: {1} Loss: {2}", e, stopwatch.ElapsedMilliseconds, loss);
                //need to calculate errors

                dataSource.Shuffle();
            }
        }

    }
}
