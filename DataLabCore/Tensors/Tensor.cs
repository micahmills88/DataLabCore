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
        public Tensor() { }

        public Tensor(float[] inputs)
            : this(1, inputs.Length, 1, 1, inputs) { }

        public Tensor(int rows, int columns, float[] inputs)
            : this(rows, columns, 1, 1, inputs) { }

        public Tensor(int rows, int columns, int layers, float[] inputs)
            : this(rows, columns, layers, 1, inputs) { }

        public Tensor(int rows, int cols, int lays, int cubes, float[] data)
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

            var controller = TensorController.Instance;
            _buffer = controller.AllocateBuffer(Size);
            SynchronizeToRemote();
            controller.Synchronize();
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

        public void SetDataView(int start, int size)
        {
            var subview = _buffer.View.GetSubView(start, size);
            DataView = subview;
        }
    }
}
