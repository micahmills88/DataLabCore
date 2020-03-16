using ILGPU;
using ILGPU.Runtime;
using System;
using System.Collections.Generic;
using System.Text;

namespace DataLabCore
{
    public class Tensor
    {
        public int Rows;
        public int Columns;
        public int Layers;
        public int Cubes;

        public int LayerSize;
        public int CubeSize;
        public int Size;

        public float[] Data;
        protected MemoryBuffer<float> _buffer;
        public ArrayView<float> DataView;

        protected bool remoteSynchronized = false;
        protected bool localSynchronized = false;

        #region constructors
        public Tensor(TensorController controller, float[] inputs)
            : this(controller, 1, inputs.Length, 1, 1, inputs) { }

        public Tensor(TensorController controller, int rows, int columns, float[] inputs)
            : this(controller, rows, columns, 1, 1, inputs) { }

        public Tensor(TensorController controller, int rows, int columns, int layers, float[] inputs)
            : this(controller, rows, columns, layers, 1, inputs) { }

        public Tensor(TensorController controller, int rows, int cols, int lays, int cubes, float[] data)
        {
            Rows = rows;
            Columns = cols;
            Layers = lays;
            Cubes = cubes;

            LayerSize = Rows * Columns;
            CubeSize = LayerSize * Layers;
            Size = CubeSize * Cubes;
            Data = data;

            if(data.Length != Size)
            {
                throw new Exception("Data length does not match Tensor size");
            }

            _buffer = controller.AllocateBuffer(Size);
            _buffer.MemSetToZero();
            SynchronizeToRemote();
        }
        #endregion

        public void SynchronizeToRemote()
        {
            //copy local data to device
            _buffer.CopyFrom(Data, 0, 0, _buffer.Extent);
            remoteSynchronized = true;
            DataView = _buffer.View;
        }

        public void SynchronizeToLocal()
        {
            //copy device data locally
            _buffer.CopyTo(Data, 0, 0, _buffer.Extent);
            localSynchronized = true;
        }

        public void Transpose2DValues()
        {
            int temp = Rows;
            Rows = Columns;
            Columns = temp;
        }

        public void SetDimensions(int rows, int columns, int layers, int cubes)
        {
            int newsize = rows * columns * layers * cubes;
            if(newsize != Size)
            {
                throw new Exception("Reshape failures, size must match");
            }
            Rows = rows;
            Columns = columns;
            Layers = layers;
            Cubes = cubes;
        }

        public void ZeroMemory()
        {
            _buffer.MemSetToZero();
        }
    }
}
