using System;
using System.Collections.Generic;
using System.Text;

namespace DataLabCore
{
    public interface IDataSource
    {
        Tensor GetSampleBatch(int batchnum);
        Tensor GetLabelBatch(int batchnum);
        int GetTotalBatches();
        void Shuffle();
    }
}
