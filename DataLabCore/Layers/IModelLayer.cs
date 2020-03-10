using System;
using System.Collections.Generic;
using System.Text;

namespace DataLabCore
{
    public interface IModelLayer
    {
        void Initialize(ILGPU.Runtime.Accelerator accelerator, Utils.RandomGenerator random);
        ILGPU.ArrayView<float> Forward(ILGPU.Runtime.Accelerator accelerator, ILGPU.ArrayView<float> inputs);
        ILGPU.ArrayView<float> Backward(ILGPU.Runtime.Accelerator accelerator, ILGPU.ArrayView<float> errors, bool calculateReturnErrors = true);
    }
}
