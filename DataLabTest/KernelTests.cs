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
                1,1,1,0,1,
                2,5,2,8,2,
                4,3,3,7,3
            };

            float[] transposed = new float[]
            {
                1,2,4,
                1,5,3,
                1,2,3,
                0,8,7,
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

        [Fact]
        public void Test_SubtractTransposed()
        {
            float[] data = new float[]
            {
                1,1,1,2,1,
                4,5,2,2,2,
                3,3,3,9,7
            };

            float[] lbl = new float[]
            {
                0.1f, 0.1f, 0.1f, 0.1f, 0.1f,
                0.2f, 0.2f, 0.2f, 0.2f, 0.2f,
                0.3f, 0.3f, 0.3f, 0.3f, 0.3f
            };

            float[] transposed = new float[]
            {
                0.9f, 3.8f, 2.7f,
                0.9f, 4.8f, 2.7f,
                0.9f, 1.8f, 2.7f,
                1.9f, 1.8f, 8.7f,
                0.9f, 1.8f, 6.7f
            };

            Tensor t1 = new Tensor(_tc, 3, 5, data);
            Tensor t2 = new Tensor(_tc, 3, 5, lbl);
            Tensor t3 = new Tensor(_tc, 5, 3, new float[data.Length]);
            _tc.SubtractTransposed(t3, t1, t2);
            t3.SynchronizeToLocal();
            Assert.Equal(transposed, t3.Data);
        }

        [Fact]
        public void Test_ForwardCorrelation()
        {
            CorrelationCalculator calc = new CorrelationCalculator();
            calc.CalculateResults();

            Tensor t1 = new Tensor(_tc, 8, 8, 3, 5, calc.inputs.Data);
            Tensor t2 = new Tensor(_tc, 3, 3, 3, 7, calc.filters.Data);
            Tensor t3 = new Tensor(_tc, 6, 6, 7, 5, new float[6 * 6 * 7 * 5]);

            _tc.ForwardCorrelation(t3, t1, t2);
            t3.SynchronizeToLocal();
            Assert.Equal(calc.results.Data, t3.Data);
        }

        [Fact]
        public void Test_BackwardConvolution()
        {
            var calc = new BackwardConvolutionCalculator();
            calc.CalculateResults();

            Tensor t1 = new Tensor(_tc, 10, 10, 7, 5, calc.errors.Data);
            Tensor t2 = new Tensor(_tc, 3, 3, 3, 7, calc.filters.Data);
            Tensor t3 = new Tensor(_tc, 8, 8, 3, 5, new float[8*8*3*5]);

            _tc.BackwardConvolution(t3, t1, t2);
            t3.SynchronizeToLocal();
            Assert.Equal(calc.results.Data, t3.Data);
        }

        [Fact]
        public void Test_WeightErrorCorrelation()
        {
            var calc = new WeightErrorCorrelationCalculator();
            calc.CalculateResults();

            Tensor t1 = new Tensor(_tc, 8, 8, 3, 5, calc.inputs.Data);
            Tensor t2 = new Tensor(_tc, 6, 6, 7, 5, calc.errors.Data);
            Tensor t3 = new Tensor(_tc, 3, 3, 3, 7, new float[3*3*3*7]);

            _tc.WeightErrorCorrelation(t3, t1, t2);
            t3.SynchronizeToLocal();
            Assert.Equal(calc.results.Data, t3.Data);
        }

        [Fact]
        public void Test_PadTensor()
        {
            var original = new float[] //3x3x3x2
            {
                1,1,1,
                1,1,1,
                1,1,1,

                2,2,2,
                2,2,2,
                2,2,2,

                3,3,3,
                3,3,3,
                3,3,3,

                4,4,4,
                4,4,4,
                4,4,4,

                5,5,5,
                5,5,5,
                5,5,5,

                6,6,6,
                6,6,6,
                6,6,6
            };

            var padded = new float[] //pad 2
            {
                0,0,0,0,0,0,0,
                0,0,0,0,0,0,0,
                0,0,1,1,1,0,0,
                0,0,1,1,1,0,0,
                0,0,1,1,1,0,0,
                0,0,0,0,0,0,0,
                0,0,0,0,0,0,0,

                0,0,0,0,0,0,0,
                0,0,0,0,0,0,0,
                0,0,2,2,2,0,0,
                0,0,2,2,2,0,0,
                0,0,2,2,2,0,0,
                0,0,0,0,0,0,0,
                0,0,0,0,0,0,0,

                0,0,0,0,0,0,0,
                0,0,0,0,0,0,0,
                0,0,3,3,3,0,0,
                0,0,3,3,3,0,0,
                0,0,3,3,3,0,0,
                0,0,0,0,0,0,0,
                0,0,0,0,0,0,0,

                0,0,0,0,0,0,0,
                0,0,0,0,0,0,0,
                0,0,4,4,4,0,0,
                0,0,4,4,4,0,0,
                0,0,4,4,4,0,0,
                0,0,0,0,0,0,0,
                0,0,0,0,0,0,0,

                0,0,0,0,0,0,0,
                0,0,0,0,0,0,0,
                0,0,5,5,5,0,0,
                0,0,5,5,5,0,0,
                0,0,5,5,5,0,0,
                0,0,0,0,0,0,0,
                0,0,0,0,0,0,0,

                0,0,0,0,0,0,0,
                0,0,0,0,0,0,0,
                0,0,6,6,6,0,0,
                0,0,6,6,6,0,0,
                0,0,6,6,6,0,0,
                0,0,0,0,0,0,0,
                0,0,0,0,0,0,0
            };

            Tensor t1 = new Tensor(_tc, 3, 3, 3, 2, original);
            Tensor t2 = new Tensor(_tc, 7, 7, 3, 2, new float[7 * 7 * 3 * 2]);
            _tc.PadTensor(t2, t1, 2, 2);
            t2.SynchronizeToLocal();
            Assert.Equal(padded, t2.Data);
        }

        [Fact]
        public void Test_InvertFilter()
        {
            var filter = new float[] //3x3x1, 3
            {
                1, 1, 0,
                3, 1, 1,
                1, 0, 2,

                2, 0, 8,
                4, 2, 0,
                9, 0, 2,

                3, 0, 2,
                0, 3, 5,
                7, 0, 3
            };

            var inverted = new float[] //3x3x1, 3
           {
                2, 0, 1,
                1, 1, 3,
                0, 1, 1,

                2, 0, 9,
                0, 2, 4,
                8, 0, 2,

                3, 0, 7,
                5, 3, 0,
                2, 0, 3
           };

            Tensor t1 = new Tensor(_tc, 3, 3, 1, 3, filter);
            Tensor t2 = new Tensor(_tc, 3, 3, 1, 3, new float[3 * 3 * 1 * 3]);
            _tc.InvertFilters(t2, t1);
            t2.SynchronizeToLocal();
            Assert.Equal(inverted, t2.Data);
        }

        [Fact]
        public void Test_MaxPool()
        {
            var data = new float[]
            {
                1,2,3,4,5,6,
                2,4,6,8,10,12,
                1,1,1,1,1,1,
                5,4,3,2,1,0,

                9,6,7,3,4,1,
                2,3,6,1,8,2,
                7,1,4,5,9,2,
                4,1,2,3,8,6
            };

            var max = new float[]
            {
                4, 8, 12,
                5, 3, 1,

                9, 7, 8,
                7, 5, 9
            };

            var mask = new float[]
            {
                0,0,0,0,0,0,
                0,1,0,1,0,1,
                0,0,0,0,1,0,
                1,0,1,0,0,0,

                1,0,1,0,0,0,
                0,0,0,0,1,0,
                1,0,0,1,1,0,
                0,0,0,0,0,0
            };

            var backward = new float[]
            {
                0,0,0,0,0,0,
                0,4,0,8,0,12,
                0,0,0,0,1,0,
                5,0,3,0,0,0,

                9,0,7,0,0,0,
                0,0,0,0,8,0,
                7,0,0,5,9,0,
                0,0,0,0,0,0
            };

            Tensor t1 = new Tensor(_tc, 4, 6, 1, 2, data);
            Tensor t2 = new Tensor(_tc, 2, 3, 1, 2, new float[2*3*1*2]);
            Tensor t3 = new Tensor(_tc, 4, 6, 1, 2, new float[4*6*1*2]);

            Tensor t4 = new Tensor(_tc, 4, 6, 1, 2, new float[4 * 6 * 1 * 2]);

            _tc.MaxPoolForward(t1, t2, t3);
            t2.SynchronizeToLocal();
            t3.SynchronizeToLocal();
            Assert.Equal(max, t2.Data);
            Assert.Equal(mask, t3.Data);

            _tc.MaxPoolBackward(t4, t3, t2);
            t4.SynchronizeToLocal();
            Assert.Equal(backward, t4.Data);
        }

        [Fact]
        public void Test_SumCubes()
        {
            var cubecalc = new CubeCalculator();
            cubecalc.CalculateResults();

            Tensor t1 = new Tensor(_tc, 8, 8, 3, 5, cubecalc.inputs.Data);
            Tensor t2 = new Tensor(_tc, 8, 8, 3, 1, new float[8*8*3]);
            _tc.SumCubes(t2, t1);
            t2.SynchronizeToLocal();

            Assert.Equal(cubecalc.results.Data, t2.Data);
        }

        [Fact]
        public void Test_Softmax()
        {
            SoftmaxCalculator sc = new SoftmaxCalculator();
            
            sc.CalculateResults();
            Tensor t1 = new Tensor(_tc, 100, 10, 1, 1, sc.values.Data);
            _tc.SoftMax(t1);
            t1.SynchronizeToLocal();

            Console.Write("test");
            Assert.Equal(sc.results.Data, t1.Data);
        }
    }
}
