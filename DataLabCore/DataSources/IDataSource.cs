using System;
using System.Collections.Generic;
using System.Text;

namespace DataLabCore
{
    public interface IDataSource
    {
        int SampleHeight { get; }
        int SampleWidth { get; }
        int SampleDepth { get; }
        int SampleSize { get; }
        int SampleCount { get; }


        void Initialize(TensorController tc, int batchSize);
        Tensor GetSampleBatch(int batchnum);
        Tensor GetLabelBatch(int batchnum);
        int GetTotalBatches();
        void Shuffle();
    }
}
