using System;
using System.Collections.Generic;
using System.Text;

namespace DataLabCore
{
    static class ClassExtensions
    {
        private static Random rng = new Random();
        public static void Shuffle<T>(this IList<T> list)
        {
            //count through the list swapping each index with a random index
            for (int i = 0; i < list.Count; i++)
            {
                int r = rng.Next(i, list.Count);
                T value = list[r];
                list[r] = list[i];
                list[i] = value;
            }
        }
    }
}
