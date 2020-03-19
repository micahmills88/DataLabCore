using System;
using System.Collections.Generic;
using System.Text;
using System.Linq;

namespace DataLabTest
{
    public class TestTensor
    {
        public int Rows;
        public int Columns;
        public int Layers;
        public int Cubes;

        public int CubeSize;
        public int LayerSize;
        public int Size;

        public float[] Data;
        public TestTensor(int rows, int columns, int layers, int cubes, float[] data)
        {
            Rows = rows;
            Columns = columns;
            Layers = layers;
            Cubes = cubes;
            Data = data;
            LayerSize = Rows * Columns;
            CubeSize = LayerSize * Layers;
            Size = CubeSize * Cubes;
        }

        public TestTensor GetCube(int cubeIndex)
        {
            var cube = new float[CubeSize];
            int offset = cubeIndex * CubeSize;
            for (int i = 0; i < CubeSize; i++)
            {
                cube[i] = Data[offset + i];
            }
            return new TestTensor(Rows, Columns, Layers, 1, cube);
        }

        public TestTensor GetLayer(int cubeIndex, int layerIndex)
        {
            var layer = new float[LayerSize];
            int offset = (cubeIndex * CubeSize) + (layerIndex * LayerSize);
            for (int i = 0; i < LayerSize; i++)
            {
                layer[i] = Data[offset + i];
            }
            return new TestTensor(Rows, Columns, 1, 1, layer);
        }

        public TestTensor GetRow(int cubeIndex, int layerIndex, int rowIndex)
        {
            var row = new float[Columns];
            int offset = (cubeIndex * CubeSize) + (layerIndex * LayerSize) + (rowIndex * Columns);
            for (int i = 0; i < Columns; i++)
            {
                row[i] = Data[offset + i];
            }
            return new TestTensor(1, Columns, 1, 1, row);
        }

        public TestTensor GetColumn(int cubeIndex, int layerIndex, int columnIndex)
        {
            var column = new float[Rows];
            int layerOffset = (cubeIndex * CubeSize) + (layerIndex * LayerSize) + columnIndex;
            for (int i = 0; i < Rows; i++)
            {
                int offset = layerOffset + (i * Columns);
                column[i] = Data[offset + i];
            }
            return new TestTensor(Rows, 1, 1, 1, column);
        }

        public TestTensor GetSubCube(int cubeIndex, int layerStart, int layers, int rowStart, int rows, int columnStart, int columns)
        {
            var result = new List<float>();
            for (int i = 0; i < layers; i++)
            {
                result.AddRange(GetSubLayer(cubeIndex, layerStart + i, rowStart, rows, columnStart, columns).Data);
            }
            return new TestTensor(rows, columns, layers, 1, result.ToArray());
        }

        public TestTensor GetSubLayer(int cubeIndex, int layerIndex, int rowStart, int rows, int columnStart, int columns)
        {
            var result = new List<float>();
            for (int i = 0; i < rows; i++)
            {
                result.AddRange(GetSubRow(cubeIndex, layerIndex, rowStart + i, columnStart, columns).Data);
            }
            return new TestTensor(rows, columns, 1, 1, result.ToArray());
        }

        public TestTensor GetSubRow(int cubeIndex, int layerIndex, int rowIndex, int start, int columns)
        {
            var row = new float[columns];
            int offset = (cubeIndex * CubeSize) + (layerIndex * LayerSize) + (rowIndex * Columns) + start;
            for (int i = 0; i < columns; i++)
            {
                row[i] = Data[offset + i];
            }
            return new TestTensor(1, columns, 1, 1, row);
        }

        public static TestTensor operator *(TestTensor left, TestTensor right)
        {
            var result = new float[left.Size];
            for (int i = 0; i < result.Length; i++)
            {
                result[i] = left.Data[i] * right.Data[i];
            }
            return new TestTensor(left.Rows, left.Columns, left.Layers, left.Cubes, result);
        }

        public float Sum()
        {
            float sum = 0f;
            foreach(float f in Data)
            {
                sum += f;
            }
            return sum;
        }

        public void SetValue(float value, int cube, int layer, int row, int column)
        {
            int offset = (cube * CubeSize) + (layer * LayerSize) + (row * Columns) + column;
            Data[offset] = value;
        }

        public TestTensor TransposeCubeLayers()
        {
            int layers = Cubes;
            int cubes = Layers;
            var result = new List<float>();
            for (int i = 0; i < Layers; i++)
            {
                for (int j = 0; j < Cubes; j++)
                {
                    result.AddRange(GetLayer(j, i).Data);
                }
            }
            return new TestTensor(Rows, Columns, layers, cubes, result.ToArray());
        }
    }
}
