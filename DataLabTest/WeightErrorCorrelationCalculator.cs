using System;
using System.Collections.Generic;
using System.Text;

namespace DataLabTest
{
    public class WeightErrorCorrelationCalculator
    {
        public TestTensor results;
        public TestTensor inputs;
        public TestTensor errors;

        public WeightErrorCorrelationCalculator()
        {
            Random rand = new Random(10);
            //generate an input of 8x8x3x5
            int inputSize = 8 * 8 * 3 * 5;
            var ins = new float[inputSize];
            for (int i = 0; i < inputSize; i++)
            {
                ins[i] = (float)rand.NextDouble();
            }
            inputs = new TestTensor(8, 8, 3, 5, ins);

            //generate errors of 6x6x7x5
            int errorSize = 6 * 6 * 7 * 5;
            var errs = new float[errorSize];
            for (int i = 0; i < errorSize; i++)
            {
                errs[i] = (float)rand.NextDouble();
            }
            errors = new TestTensor(6, 6, 7, 5, errs);

            //results should be 3x3x3x7
            int resultSize = 3 * 3 * 3 * 7;
            var res = new float[resultSize];
            results = new TestTensor(3, 3, 3, 7, res);
        }

        public void CalculateResults()
        {
            var errorCubes = errors.TransposeCubeLayers();
            var inputCubes = inputs.TransposeCubeLayers();
            for (int i = 0; i < results.Cubes; i++)
            {
                var errorCube = errorCubes.GetCube(i);
                for (int j = 0; j < results.Layers; j++)
                {
                    var inputCube = inputCubes.GetCube(j);
                    for (int k = 0; k < results.Rows; k++)
                    {
                        for (int l = 0; l < results.Columns; l++)
                        {
                            var multiple = inputCube.GetSubCube(0, 0, inputCube.Layers, k, errorCube.Rows, l, errorCube.Columns) * errorCube;
                            var sum = multiple.Sum();
                            results.SetValue(sum, i, j, k, l);
                        }
                    }
                }
            }
        }
    }
}
