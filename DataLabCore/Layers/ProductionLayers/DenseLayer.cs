using System;
using System.Collections.Generic;
using System.Text;

namespace DataLabCore
{
    public class DenseLayer : IProductionLayer
    {
        TensorController _controller;

        int _input_count;
        int _output_count;
        int _batch_size;
        ActivationType _activation;

        public int OutputHeight { get => 1; }
        public int OutputWidth { get => 1; }
        public int OutputDepth { get => 1; }
        public int OutputSize { get => _output_count; }

        public Tensor Forward(Tensor data)
        {
            throw new NotImplementedException();
        }
    }
}
