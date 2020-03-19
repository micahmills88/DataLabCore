using System;
using System.Collections.Generic;
using System.Text;

namespace DataLabTest
{
    public class CorrelationCalculator
    {
        public TestTensor inputs;
        public TestTensor results;
        public TestTensor filters;

        public CorrelationCalculator()
        {
            Random rand = new Random(10);
            //generate an input of 8x8x3x5
            int inputSize = 8 * 8 * 3 * 5;
            var ins = new float[inputSize];
            for (int i = 0; i < ins.Length; i++)
            {
                ins[i] = (float)rand.NextDouble();
            }
            inputs = new TestTensor(8, 8, 3, 5, ins);

            //generate an filters of 3x3x3x7
            int filterSize = 3 * 3 * 3 * 7;
            var filts = new float[filterSize];
            for (int i = 0; i < filts.Length; i++)
            {
                filts[i] = (float)rand.NextDouble();
            }
            filters = new TestTensor(3, 3, 3, 7, filts);

            //results should be 6x6x7x5
            int resultSize = 6 * 6 * 7 * 5;
            var res = new float[resultSize];
            results = new TestTensor(6, 6, 7, 5, res);
        }

        public void CalculateResults()
        {
            for (int i = 0; i < results.Cubes; i++)
            {
                var inputCube = inputs.GetCube(i);
                for (int j = 0; j < results.Layers; j++)
                {
                    var filter = filters.GetCube(j);
                    for (int k = 0; k < results.Rows; k++)
                    {
                        for (int l = 0; l < results.Columns; l++)
                        {
                            var sum = (inputCube.GetSubCube(0, 0, inputCube.Layers, k, filter.Rows, l, filter.Columns) * filter).Sum();
                            results.SetValue(sum, i, j, k, l);
                        }
                    }
                }
            }
        }
    }
}
