using System;
using System.Collections.Generic;
using System.Text;

namespace DataLabTest
{
    public class BackwardConvolutionCalculator
    {
        public TestTensor errors; //pretend these are padded
        public TestTensor filters; //pretend these are inverted
        public TestTensor results;

        public BackwardConvolutionCalculator()
        {
            Random rand = new Random(10);
            //"padded" errors of 10x10x7x5
            int errorSize = 10 * 10 * 7 * 5;
            var err = new float[errorSize];
            for (int i = 0; i < errorSize; i++)
            {
                err[i] = (float)rand.NextDouble();
            }
            errors = new TestTensor(10, 10, 7, 5, err);

            //generate filters of 3x3x3x7
            int filterSize = 3 * 3 * 3 * 7;
            var filt = new float[filterSize];
            for (int i = 0; i < filterSize; i++)
            {
                filt[i] = (float)rand.NextDouble();
            }
            filters = new TestTensor(3, 3, 3, 7, filt);

            //results should be 8x8x3x5
            int resultSize = 8 * 8 * 3 * 5;
            var res = new float[resultSize];
            results = new TestTensor(8, 8, 3, 5, res);
        }

        public void CalculateResults()
        {
            var filtersByLayer = filters.TransposeCubeLayers();
            for (int i = 0; i < results.Cubes; i++)
            {
                var errorCube = errors.GetCube(i);
                for (int j = 0; j < results.Layers; j++)
                {
                    var filterCube = filtersByLayer.GetCube(j);
                    for (int k = 0; k < results.Rows; k++)
                    {
                        for (int l = 0; l < results.Columns; l++)
                        {
                            var multiple = errorCube.GetSubCube(0, 0, errorCube.Layers, k, filterCube.Rows, l, filterCube.Columns) * filterCube;
                            var sum = multiple.Sum();
                            results.SetValue(sum, i, j, k, l);
                        }
                    }
                }
            }
        }
    }
}
