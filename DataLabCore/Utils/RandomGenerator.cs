using System;
using System.Collections.Generic;
using System.Text;

namespace DataLabCore
{
    public static class RandomGenerator
    {
        private static Random random = new Random(125);

        //gets a float from a gaussian distribution using box meuller
        public static float GetFloatN()
        {
            return GetFloatN(0f, 1.0f);
        }

        public static float GetFloatN(float mean, float deviation)
        {
            //mean of 0 and std deviation of 1
            //matches test in mathnet random
            double u1 = 1.0 - random.NextDouble();
            double u2 = 1.0 - random.NextDouble();
            double randStdNormal = Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Sin(2.0 * Math.PI * u2);
            return (float)(randStdNormal * deviation + mean);
        }

        public static float GetFloat(float lower, float upper)
        {
            double d = random.NextDouble() * (upper - lower) + lower;
            return (float)d;
        }

        public static float[] GetFloatNormalDistribution(int count, float mean, float deviation)
        {
            float[] out_data = new float[count];
            for (int i = 0; i < count; i++)
            {
                out_data[i] = GetFloatN(mean, deviation);
            }
            return out_data;
        }

        public static float[] GetFloatUniformDistribution(int count, float lower, float upper)
        {
            float[] out_data = new float[count];
            for (int i = 0; i < count; i++)
            {
                out_data[i] = GetFloat(lower, upper);
            }
            return out_data;
        }
    }
}
