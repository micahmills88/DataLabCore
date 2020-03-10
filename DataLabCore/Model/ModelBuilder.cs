using System;
using System.Collections.Generic;
using System.Text;
using System.Linq;

using ILGPU;
using ILGPU.Runtime;
using ILGPU.Runtime.Cuda;

using DataLabCore.Utils;

namespace DataLabCore
{
    public partial class ModelBuilder
    {
        Context _context;
        Accelerator _accelerator;

        RandomGenerator _random = new RandomGenerator();
        List<IModelLayer> _layers = new List<IModelLayer>();

        public ModelBuilder()
        {
            _context = new Context();
            _accelerator = new CudaAccelerator(_context);
        }

        public void AddLayer(IModelLayer layer)
        {
            layer.Initialize(_accelerator, _random);
            _layers.Add(layer);
        }

        public void FitModel(DataSource dataSource, LossCalculator lossCalculator)
        {
            dataSource.Initialize(_accelerator);
            lossCalculator.Initialize(_accelerator);

            for (int i = 0; i < dataSource.BatchCount; i++)
            {
                var data = dataSource.GetSamples(i);
                for (int l = 0; l < _layers.Count; l++)
                {
                    var layer = _layers[l];
                    data = layer.Forward(_accelerator, data);
                }
                var errors = lossCalculator.CalculateLoss(_accelerator, data, dataSource.GetLabels(i));
                for (int l = _layers.Count - 1; l >= 0; l--)
                {
                    var layer = _layers[l];
                    layer.Backward(_accelerator, errors);
                    //errors = layer.Errors;
                }
            }
        }
    }
}
