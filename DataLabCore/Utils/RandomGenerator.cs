using System;
using System.Collections.Generic;
using System.Text;

namespace DataLabCore.Utils
{
    public class RandomGenerator
    {
        Random random;

        public RandomGenerator()
        {
            random = new Random();
        }

        public RandomGenerator(int seed)
        {
            random = new Random(seed);
        }

        //gets a float from a gaussian distribution using box meuller
        public float GetFloatN()
        {
            return GetFloatN(0f, 1.0f);
        }

        public float GetFloatN(float mean, float deviation)
        {
            //mean of 0 and std deviation of 1
            //matches test in mathnet random
            double u1 = 1.0 - random.NextDouble();
            double u2 = 1.0 - random.NextDouble();
            double randStdNormal = Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Sin(2.0 * Math.PI * u2);
            return (float)(randStdNormal * deviation + mean);
        }

        public float[] GetFloatDistribution(int count, float mean, float deviation)
        {
            float[] out_data = new float[count];
            for (int i = 0; i < count; i++)
            {
                out_data[i] = GetFloatN(mean, deviation);
            }
            return out_data;
        }
    }
}
