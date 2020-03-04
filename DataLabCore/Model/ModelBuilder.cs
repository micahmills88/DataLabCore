using System;
using System.Collections.Generic;
using System.Text;
using ILGPU;
using ILGPU.Runtime;
using ILGPU.Runtime.Cuda;

namespace DataLabCore
{
    public class ModelBuilder
    {
        private Context _context;
        private Accelerator _accelerator;



        public ModelBuilder()
        {
            _context = new Context();
            _accelerator = new CudaAccelerator(_context);
        }

        public void AddLayer(LayerType layerType, int inputs, int outputs, ActivationType activationType)
        {
            var layer = new ModelLayer(_accelerator, layerType, inputs, outputs, activationType);
        }

        public void FitModel(DataSource dataSource, LossFunction lossFunction, int batchSize = 10, float learningRate = 1.0f)
        {

        }
    }
}
