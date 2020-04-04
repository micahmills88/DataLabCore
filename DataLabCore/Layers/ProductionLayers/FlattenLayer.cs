using System;
using System.Collections.Generic;
using System.Text;

namespace DataLabCore
{
    public class FlattenLayer : IProductionLayer
    {
        TensorController _controller;
        int _input_rows;
        int _input_columns;
        int _input_layers;
        int _input_cubes;
        int _output_size;

        public int OutputHeight { get => 1; }
        public int OutputWidth { get => 1; }
        public int OutputDepth { get => 1; }
        public int OutputSize { get => _output_size; }

        public Tensor Forward(Tensor data)
        {
            throw new NotImplementedException();
        }
    }
}
