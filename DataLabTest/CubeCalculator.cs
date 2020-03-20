using System;
using System.Collections.Generic;
using System.Text;

namespace DataLabTest
{
    public class CubeCalculator
    {
        public TestTensor inputs;
        public TestTensor results;

        public CubeCalculator()
        {
            Random rand = new Random(10);
            int inputSize = 8 * 8 * 3 * 5;
            var ins = new float[inputSize];
            for (int i = 0; i < ins.Length; i++)
            {
                ins[i] = (float)rand.NextDouble();
            }
            inputs = new TestTensor(8, 8, 3, 5, ins);

            int resultSize = 8 * 8 * 3;
            var res = new float[resultSize];
            results = new TestTensor(8, 8, 3, 1, res);
        }

        public void CalculateResults()
        {
            for (int i = 0; i < inputs.Cubes; i++)
            {
                var cube = inputs.GetCube(i);
                results = results + cube;
            }
        }
    }
}
