using System;
using System.Collections.Generic;
using System.Text;

namespace DataLabTest
{
    public class SoftmaxCalculator
    {
        public TestTensor values;
        public TestTensor results;

        public SoftmaxCalculator()
        {
            Random rand = new Random(10);
            int valSize = 100 * 10;
            var vals = new float[valSize];
            for (int i = 0; i < valSize; i++)
            {
                vals[i] = (float)rand.NextDouble();
            }
            values = new TestTensor(100, 10, 1, 1, vals);

            int resultSize = 100*10;
            var res = new float[resultSize];
            results = new TestTensor(100, 10, 1, 1, res);
        }

        public void CalculateResults()
        {
            var temp = values.GetCube(0);
            for (int i = 0; i < values.Size; i++)
            {
                temp.Data[i] = (float)Math.Exp(temp.Data[i]);
            }

            var rows = new float[100];
            for (int i = 0; i < 100; i++)
            {
                rows[i] = temp.GetRow(0, 0, i).Sum();
            }

            for (int i = 0; i < values.Size; i++)
            {
                int idx = i / 10;
                results.Data[i] = temp.Data[i] / rows[idx];
            }
        }


    }
}
