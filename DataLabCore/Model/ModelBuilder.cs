﻿using System;
using System.Collections.Generic;
using System.Text;
using System.Linq;

using ILGPU;
using ILGPU.Runtime;
using ILGPU.Runtime.Cuda;


namespace DataLabCore
{
    public class ModelBuilder
    {
        List<IModelLayer> _layers = new List<IModelLayer>();

        public ModelBuilder()
        {

        }

        public void AddLayer(IModelLayer layer)
        {
            _layers.Add(layer);
        }

        public void FitModel(DataSource dataSource, int epochs)
        {
            for (int e = 0; e < epochs; e++)
            {
                for (int b = 0; b < dataSource.TotalBatches; b++)
                {
                    var data = dataSource.GetSampleBatch(b);
                    for (int i = 0; i < _layers.Count; i++)
                    {
                        data = _layers[i].Forward(data);
                    }
                    var error = dataSource.GetLabelBatch(b);
                    for (int i = _layers.Count - 1; i >= 0; i--)
                    {
                        error = _layers[i].Backward(error, i > 0);
                    }
                }
                dataSource.Shuffle();
            }
        }

    }
}
