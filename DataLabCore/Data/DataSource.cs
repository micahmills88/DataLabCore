using System;
using System.Collections.Generic;
using System.Text;

using ILGPU;
using ILGPU.Runtime;

namespace DataLabCore
{
    public class DataSource
    {
        public int BatchCount;

        public DataSource(int batchSize)
        {

        }

        public void Initialize(Accelerator _accelerator)
        {
            //load data into memory
        }

        public ArrayView<float> GetSamples(int batchNum)
        {
            throw new NotImplementedException();
        }

        public ArrayView<float> GetLabels(int batchNum)
        {
            throw new NotImplementedException();
        }
    }
}
