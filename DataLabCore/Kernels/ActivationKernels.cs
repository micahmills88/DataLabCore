using ILGPU;
using ILGPU.Algorithms;
using System;
using System.Collections.Generic;
using System.Text;

namespace DataLabCore.Kernels
{
    public class ActivationKernels
    {
        #region activations
        private void ActivateSigmoid(ILGPU.Index index, ArrayView<float> outputs)
        {
            outputs[index] = 1.0f / (1.0f + (float)XMath.Exp(-outputs[index]));
        }
        #endregion activations

        #region derivatives
        private void DeriveSigmoid(ILGPU.Index index, ArrayView<float> outputs)
        {
            outputs[index] = outputs[index] * (1.0f - outputs[index]);
        }
        #endregion derivatives
    }
}
