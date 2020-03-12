using System;
using System.Collections.Generic;
using System.IO;
using System.Text;

namespace DataLabCore
{
    public class DataSource
    {
        int _width = 28;
        int _height = 28;
        int _depth = 1;
        int _classes = 10;
        int _samplecount = 60000;
        int _batchsize = 10;

        public int TotalBatches { get {  return _samplecount / _batchsize; } }

        string samplePath = @"F:\Machine_Learning\Datasets\MNIST\train-images.idx3-ubyte";
        string labelPath = @"F:\Machine_Learning\Datasets\MNIST\train-labels.idx1-ubyte";

        List<string> keys = new List<string>();
        Dictionary<string, float[]> data_labels = new Dictionary<string, float[]>();
        Dictionary<string, float[]> data_samples = new Dictionary<string, float[]>();

        Tensor sampleTensor;
        Tensor labelTensor;

        public DataSource(int batchSize)
        {
            this._batchsize = batchSize;
            BinaryReader training_images = new BinaryReader(new FileStream(samplePath, FileMode.Open));
            BinaryReader training_labels = new BinaryReader(new FileStream(labelPath, FileMode.Open));

            training_images.BaseStream.Seek(16, SeekOrigin.Begin);
            training_labels.BaseStream.Seek(8, SeekOrigin.Begin);

            Console.WriteLine("Loading samples for training...");
            for (int i = 0; i < _samplecount; i++)
            {
                byte[] imageBytes = training_images.ReadBytes(_width * _height * _depth);
                int labelByte = training_labels.ReadByte();
                float[] sample = Array.ConvertAll(imageBytes, item => (float)(item / 255f));
                float[] label = new float[_classes];
                label[labelByte] = 1.0f;
                string guid = Guid.NewGuid().ToString("N");
                keys.Add(guid);
                data_labels.Add(guid, label);
                data_samples.Add(guid, sample);
            }

            BuildTensors();
        }

        public Tensor GetSampleBatch(int batchnum)
        {
            int offset = _width * _height * _batchsize;
            int start = batchnum * offset;
            sampleTensor.SetDataView(start, offset);
            sampleTensor.Size = offset;
            sampleTensor.Rows = _batchsize;
            sampleTensor.Columns = _width * _height;
            return sampleTensor;
        }

        public Tensor GetLabelBatch(int batchnum)
        {
            int offset = _classes * _batchsize;
            int start = batchnum * offset;
            labelTensor.SetDataView(start, offset);
            labelTensor.Size = offset;
            labelTensor.Rows = _batchsize;
            labelTensor.Columns = _classes;
            return labelTensor;
        }

        public void Shuffle()
        {
            keys.Shuffle();
            List<float> samples = new List<float>();
            List<float> labels = new List<float>();
            foreach (var key in keys)
            {
                samples.AddRange(data_samples[key]);
                labels.AddRange(data_labels[key]);
            }

            sampleTensor.Data = samples.ToArray();
            labelTensor.Data = labels.ToArray();

            sampleTensor.SynchronizeToRemote();
            labelTensor.SynchronizeToRemote();

            sampleTensor.Rows = _batchsize;
            labelTensor.Rows = _batchsize;
        }

        private void BuildTensors()
        {
            List<float> samples = new List<float>();
            List<float> labels = new List<float>();

            foreach (var key in keys)
            {
                samples.AddRange(data_samples[key]);
                labels.AddRange(data_labels[key]);
            }

            sampleTensor = new Tensor(_samplecount, _width * _height, samples.ToArray());
            labelTensor = new Tensor(_samplecount, _classes, labels.ToArray());
        }
    }
}
