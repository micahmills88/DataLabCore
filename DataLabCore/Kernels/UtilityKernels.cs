using System;
using System.Collections.Generic;
using System.Text;

namespace DataLabCore.Kernels
{
    public class UtilityKernels
    {
        #region utility kernels
        private void Transpose2D(ILGPU.Index index)
        {
            /*
         * gpu thread per new row
            so if we have a 100x256 and we need it transposed
            (index, inputs, outputs, rows, columns)
            {
	            int newrowoffset = index * rows;
	            for(int i = 0; i < rows; i++)
	            {
		            outputs[newrowoffset + i] = inputs[index + i * columns]
	            }
            }
         */
        }
        #endregion utility kernels
    }
}
