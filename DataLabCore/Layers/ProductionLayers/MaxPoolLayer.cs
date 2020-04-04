using System;
using System.Collections.Generic;
using System.Text;

namespace DataLabCore
{
    public class MaxPoolLayer : IProductionLayer
    {
        TensorController _controller;

        int _input_height;
        int _input_width;
        int _input_depth;
        int _batch_size;

        int _output_height;
        int _output_width;
        int _output_depth;

        public int OutputHeight { get => _output_height; }
        public int OutputWidth { get => _output_width; }
        public int OutputDepth { get => _output_depth; }
        public int OutputSize { get => _output_height * _output_width * _output_depth; }

        public Tensor Forward(Tensor data)
        {
            throw new NotImplementedException();
        }
    }
}
