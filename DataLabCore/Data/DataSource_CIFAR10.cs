using System;
using System.Collections.Generic;
using System.IO;
using System.Text;

namespace DataLabCore
{
    public class DataSource_CIFAR10 : IDataSource
    {
        TensorController _controller;
        int _width = 32;
        int _height = 32;
        int _sample_size = 32 * 32 * 3;
        int _depth = 3;
        int _classes = 10;
        int _samplecount = 50000;
        int _batchsize = 10;

        public int TotalBatches { get { return _samplecount / _batchsize; } }

        string[] samplePaths = new string[] {
            @"F:\Machine_Learning\Datasets\cifar-10-batches-bin\data_batch_1.bin",
            @"F:\Machine_Learning\Datasets\cifar-10-batches-bin\data_batch_2.bin",
            @"F:\Machine_Learning\Datasets\cifar-10-batches-bin\data_batch_3.bin",
            @"F:\Machine_Learning\Datasets\cifar-10-batches-bin\data_batch_4.bin",
            @"F:\Machine_Learning\Datasets\cifar-10-batches-bin\data_batch_5.bin"
        };

        List<string> keys = new List<string>();
        Dictionary<string, float[]> data_labels = new Dictionary<string, float[]>();
        Dictionary<string, float[]> data_samples = new Dictionary<string, float[]>();

        List<Tensor> sample_tensors = new List<Tensor>();
        List<Tensor> label_tensors = new List<Tensor>();

        public DataSource_CIFAR10(TensorController controller, int batchSize)
        {
            _controller = controller;
            _batchsize = batchSize;
            foreach(var samplePath in samplePaths)
            {
                BinaryReader training_data = new BinaryReader(new FileStream(samplePath, FileMode.Open));
                Console.WriteLine("Loading samples from {0}", samplePath);
                for (int i = 0; i < 10000; i++)
                {
                    int labelByte = training_data.ReadByte();
                    byte[] imageBytes = training_data.ReadBytes(_width * _height * _depth);
                    float[] sample = Array.ConvertAll(imageBytes, item => (float)(item / 255f));
                    float[] label = new float[_classes];
                    label[labelByte] = 1.0f;
                    string guid = Guid.NewGuid().ToString("N");
                    keys.Add(guid);
                    data_labels.Add(guid, label);
                    data_samples.Add(guid, sample);
                }
            }

            BuildTensors();
            SetTensorData();
        }

        public Tensor GetSampleBatch(int batchnum)
        {
            return sample_tensors[batchnum];
        }

        public Tensor GetLabelBatch(int batchnum)
        {
            return label_tensors[batchnum];
        }

        public int GetTotalBatches()
        {
            return _samplecount / _batchsize;
        }

        public void Shuffle()
        {
            keys.Shuffle();
            SetTensorData();
        }

        private void BuildTensors()
        {
            int batchCount = _samplecount / _batchsize;
            int s_size = _batchsize * _sample_size;
            int l_size = _batchsize * _classes;
            for (int i = 0; i < batchCount; i++)
            {
                sample_tensors.Add(new Tensor(_controller, _width, _height, _depth, _batchsize, new float[s_size]));
                label_tensors.Add(new Tensor(_controller, _batchsize, _classes, new float[l_size]));
            }
        }

        private void SetTensorData()
        {
            int batchCount = _samplecount / _batchsize;
            for (int i = 0; i < batchCount; i++)
            {
                List<float> tempSample = new List<float>();
                List<float> tempLabel = new List<float>();
                for (int j = 0; j < _batchsize; j++)
                {
                    int keyIdx = i * _batchsize + j;
                    var key = keys[keyIdx];
                    tempSample.AddRange(data_samples[key]);
                    tempLabel.AddRange(data_labels[key]);
                }
                sample_tensors[i].Data = tempSample.ToArray();
                sample_tensors[i].SynchronizeToRemote();
                label_tensors[i].Data = tempLabel.ToArray();
                label_tensors[i].SynchronizeToRemote();
            }
        }
    }
}
