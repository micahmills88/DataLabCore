using System;
using Xunit;
using DataLabCore;

namespace DataLabTest
{
    public class KernelTests
    {
        private readonly TensorController _tc;
        public KernelTests()
        {
            _tc = new TensorController(ControllerType.CPU);
        }

        [Fact]
        public void Test_MatrixMultiply()
        {
            float[] ma = new float[]
            {
                3,4,2,
                6,5,7
            };

            float[] mb = new float[]
            {
                -2,3,
                6,4,
                1,-1
            };

            float[] mc = new float[]
            {
                20, 23,
                25, 31
            };

            Tensor t1 = new Tensor(_tc, 2, 3, ma);
            Tensor t2 = new Tensor(_tc, 3, 2, mb);
            Tensor t3 = new Tensor(_tc, 2, 2, new float[4]);

            _tc.MatrixMultiply(t3, t1, t2);
            t3.SynchronizeToLocal();
            Assert.Equal(mc, t3.Data);
        }

        [Fact]
        public void Test_AddBias()
        {
            float[] bias = new float[]
            {
                0.1f, 0.2f, 0.3f, 0.4f
            };

            float[] matrix = new float[]
            {
                1, 2, 3, 4,
                5, 6, 7, 8,
                9, 10, 11, 12
            };

            float[] result = new float[]
            {
                1.1f, 2.2f, 3.3f, 4.4f,
                5.1f, 6.2f, 7.3f, 8.4f,
                9.1f, 10.2f, 11.3f, 12.4f
            };

            Tensor t1 = new Tensor(_tc, bias);
            Tensor t2 = new Tensor(_tc, 3, 4, matrix);

            _tc.AddBias(t2, t1);
            t2.SynchronizeToLocal();
            Assert.Equal(result, t2.Data);
        }

        [Fact]
        public void Test_Transpose2D()
        {
            float[] input = new float[]
            {
                1,1,1,1,1,
                2,2,2,2,2,
                3,3,3,3,3
            };

            float[] transposed = new float[]
            {
                1,2,3,
                1,2,3,
                1,2,3,
                1,2,3,
                1,2,3
            };

            Tensor t1 = new Tensor(_tc, 3, 5, input);
            Tensor t2 = new Tensor(_tc, 3, 5, new float[input.Length]);
            _tc.Transpose2D(t2, t1);
            t2.SynchronizeToLocal();
            Assert.Equal(transposed, t2.Data);
        }

        [Fact]
        public void Test_TransposedMatrixMultiply()
        {
            float[] ma = new float[]
            {
                3,4,2,
                6,5,7
            };

            float[] mb = new float[]
            {
                -2,3,
                6,4,
                1,-1
            };

            float[] mc = new float[]
            {
                20, 23,
                25, 31
            };

            Tensor t1 = new Tensor(_tc, 2, 3, ma);
            Tensor t2 = new Tensor(_tc, 3, 2, mb);
            Tensor t3 = new Tensor(_tc, 2, 2, new float[4]);

            _tc.TransposedMatrixMultiply(t3, t2, t1);
            t3.SynchronizeToLocal();
            Assert.Equal(mc, t3.Data);
        }

        [Fact]
        public void Test_RowSums()
        {
            float[] values = new float[]
            {
                1,1,1,1,
                2,2,2,2,
                3,3,3,3,
                4,4,4,4,
                5,5,5,5
            };

            float[] expected = new float[]
            {
                4, 8, 12, 16, 20
            };

            Tensor t1 = new Tensor(_tc, 5, 4, values);
            Tensor t2 = new Tensor(_tc, new float[expected.Length]);
            _tc.RowSum(t2, t1);
            t2.SynchronizeToLocal();
            Assert.Equal(expected, t2.Data);
        }
    }
}
