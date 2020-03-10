using ILGPU;
using ILGPU.Runtime;
using ILGPU.Runtime.Cuda;
using System;
using System.Collections.Generic;
using System.Text;

namespace DataLabCore
{
    public class TensorController
    {
        private static readonly TensorController _controllerInstance = new TensorController();
        private static Accelerator _accelerator;

        private TensorController()
        {
            var context = new Context();
            context.EnableAlgorithms();
            _accelerator = new CudaAccelerator(context);
        }

        public static TensorController Instance { get { return _controllerInstance; } }


        public void Synchronize()
        {
            _accelerator.Synchronize();
        }

        public MemoryBuffer<float> AllocateBuffer(int size)
        {
            return _accelerator.Allocate<float>(1);
        }


    }
}
