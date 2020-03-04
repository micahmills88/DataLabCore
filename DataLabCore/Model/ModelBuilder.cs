using System;
using System.Collections.Generic;
using System.Text;
using DataLabCore.Utils;
using ILGPU;
using ILGPU.Runtime;
using ILGPU.Runtime.Cuda;

namespace DataLabCore
{
    public class ModelBuilder
    {
        private Context _context;
        private Accelerator _accelerator;

        private RandomGenerator _random = new RandomGenerator();
        private List<ModelLayer> _layers = new List<ModelLayer>();

        public ModelBuilder()
        {
            _context = new Context();
            _accelerator = new CudaAccelerator(_context);
        }

        public void AddLayer(LayerType layerType, int inputs, int outputs, ActivationType activationType)
        {
            var layer = new ModelLayer(layerType, inputs, outputs, activationType);
            layer.Initialize(_accelerator, _random);
            _layers.Add(layer);
        }

        public void FitModel(DataSource dataSource, LossFunction lossFunction, int batchSize = 10, float learningRate = 1.0f)
        {

        }
    }
}
