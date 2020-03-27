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

        LossLayer _loss_layer;

        public ModelBuilder()
        {

        }

        public void AddLayer(IModelLayer layer)
        {
            _layers.Add(layer);
        }

        public void AddLossLayer(LossLayer losslayer)
        {
            _loss_layer = losslayer;
        }

        public void FitModel(IDataSource dataSource, int epochs, float lossLimit, float learningRate)
        {
            if(_loss_layer == null)
            {
                throw new Exception("no loss layer");
            }

            float lowest_loss = 10f;
            for (int e = 0; e < epochs || (lowest_loss < lossLimit); e++)
            {
                
                stopwatch.Restart();
                float percent = 0f;
                Console.CursorVisible = false;
                Console.Write("Epoch: {0} ... %{1:N0}", e, percent);
                var batches = dataSource.GetTotalBatches();
                for (int b = 0; b < batches; b++)
                {
                    var data = dataSource.GetSampleBatch(b);
                    var labels = dataSource.GetLabelBatch(b);
                    for (int i = 0; i < _layers.Count; i++)
                    {
                        data = _layers[i].Forward(data);
                    }
                    var errors = _loss_layer.CalculateLoss(data, labels);
                    for (int i = _layers.Count - 1; i >= 0; i--)
                    {
                        errors = _layers[i].Backward(errors, learningRate, i > 0);
                    }
                    if (b % 25 == 0)
                    {
                        percent = 100f * (b + 1f) / (float)batches;
                        Console.SetCursorPosition(0, Console.CursorTop);
                        Console.Write("Epoch: {0} ... %{1:N0}", e, percent);
                    }
                }

                Console.CursorVisible = true;
                var loss = _loss_layer.GetEpochLoss();
                if(loss < lowest_loss)
                {
                    lowest_loss = loss;
                }
                stopwatch.Stop();
                Console.SetCursorPosition(0, Console.CursorTop);
                Console.WriteLine("Epoch: {0:D3} Time: {1} Loss: {2}", e, stopwatch.ElapsedMilliseconds, loss);
                dataSource.Shuffle();
            }
        }

    }
}
