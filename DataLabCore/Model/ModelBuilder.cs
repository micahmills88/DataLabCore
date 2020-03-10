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

    }
}
