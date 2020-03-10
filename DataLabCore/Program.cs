using System;
using System.Collections.Generic;
using System.IO;

namespace DataLabCore
{
    class Program
    {
        static void Main(string[] args)
        {
            ModelBuilder builder = new ModelBuilder();
            builder.AddLayer(new DenseLayer(784, 1, 10, ActivationType.Sigmoid));

        }
    }
}
