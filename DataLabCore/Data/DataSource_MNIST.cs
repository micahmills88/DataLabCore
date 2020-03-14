using System;
using System.Collections.Generic;
using System.IO;
using System.Text;

namespace DataLabCore
{
    public class DataSource_MNIST : IDataSource
    {
        TensorController _controller;
        int _width = 28;
        int _height = 28;
        int _sample_size = 28 * 28;
        int _depth = 1;
        int _classes = 10;
        int _samplecount = 60000;
        int _batchsize = 10;

        string samplePath = @"F:\Machine_Learning\Datasets\MNIST\train-images.idx3-ubyte";
        string labelPath = @"F:\Machine_Learning\Datasets\MNIST\train-labels.idx1-ubyte";

        List<string> keys = new List<string>();
        Dictionary<string, float[]> data_labels = new Dictionary<string, float[]>();
        Dictionary<string, float[]> data_samples = new Dictionary<string, float[]>();

        List<Tensor> sample_tensors = new List<Tensor>();
        List<Tensor> label_tensors = new List<Tensor>();

        public DataSource_MNIST(TensorController controller, int batchSize)
        {
            _controller = controller;
            _batchsize = batchSize;
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
                sample_tensors.Add(new Tensor(_controller, _batchsize, _sample_size, new float[s_size]));
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
