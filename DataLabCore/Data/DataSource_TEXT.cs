using System;
using System.Collections.Generic;
using System.Text;
using System.IO;
using System.Linq;

namespace DataLabCore
{
    public class DataSource_TEXT : IDataSource
    {
        TensorController _controller;
        int _width;         //number of chars per sample
        int _height;        //number of unique chars
        int _sample_size;   //height * width
        int _depth = 1;
        int _classes;       //same as height
        int _samplecount;   //full size -  width
        int _real_samplecount;
        int _batchsize;

        List<char> _raw_data = new List<char>();
        List<char> _unique_chars = new List<char>();

        public int SampleHeight { get => _height; }
        public int SampleWidth { get => _width; }
        public int SampleDepth { get => _depth; }
        public int SampleSize { get => _sample_size; }
        public int SampleCount { get => _samplecount; }

        //string samplePath = @"F:\Machine_Learning\Datasets\shakespear\shakespear_long_cleaned.txt";
        string samplePath = @"F:\Machine_Learning\Datasets\shakespear\shakespear_short_cleaned.txt";

        List<string> _keys = new List<string>();
        Dictionary<string, char> data_labels = new Dictionary<string, char>();
        Dictionary<string, char[]> data_samples = new Dictionary<string, char[]>();

        List<Tensor> sample_tensors = new List<Tensor>();
        List<Tensor> label_tensors = new List<Tensor>();

        public DataSource_TEXT(int charsPerSample, int samplesPerEpoch)
        {
            Console.WriteLine("Loading text from: {0}", samplePath);
            _height = charsPerSample;

            string fullText = File.ReadAllText(samplePath);
            _raw_data.AddRange(fullText.ToCharArray());
            _unique_chars.AddRange(fullText.Distinct().OrderBy(c => c));
            _width = _unique_chars.Count;
            _classes = _width;
            _sample_size = _width * _height;
            _real_samplecount = _raw_data.Count - charsPerSample;
            _samplecount = _real_samplecount;
            if (samplesPerEpoch <= _real_samplecount)
            {
                _samplecount = samplesPerEpoch;
            }

            for (int i = 0; i < _real_samplecount; i++)
            {
                var g = Guid.NewGuid().ToString("N");
                _keys.Add(g);
                var sample = _raw_data.GetRange(i, _height).ToArray();
                char label = _raw_data[i + _height];
                data_samples.Add(g, sample);
                data_labels.Add(g, label);
            }

            Console.WriteLine("Data Loaded...");
        }

        public void Initialize(TensorController tc, int batchSize)
        {
            _controller = tc;
            _batchsize = batchSize;
            BuildTensors();
            SetTensorData();
        }

        public Tensor GetSampleBatch(int batchnum)
        {
            //batch is verticle
            //1 hot encoded vectors
            //each row is a single character
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
            _keys.Shuffle();
            SetTensorData();
        }

        private void BuildTensors()
        {
            int batchCount = _samplecount / _batchsize;
            int s_size = _batchsize * _sample_size;
            int l_size = _batchsize * _classes;
            for (int i = 0; i < batchCount; i++)
            {
                sample_tensors.Add(new Tensor(_controller, _height, _width, _depth, _batchsize, new float[s_size]));
                label_tensors.Add(new Tensor(_controller, _batchsize, _classes, new float[l_size]));
            }
        }

        private void SetTensorData()
        {
            var keys = _keys.GetRange(0, _samplecount);
            int batchCount = _samplecount / _batchsize;
            for (int i = 0; i < batchCount; i++)
            {
                List<float> tempSample = new List<float>();
                List<float> tempLabel = new List<float>();
                for (int j = 0; j < _batchsize; j++)
                {
                    //foreach char in sample
                    int keyIdx = i * _batchsize + j;
                    var key = keys[keyIdx];
                    tempSample.AddRange(CharToVect(data_samples[key]));
                    tempLabel.AddRange(CharToVect(data_labels[key]));
                }
                sample_tensors[i].Data = tempSample.ToArray();
                sample_tensors[i].SynchronizeToRemote();
                label_tensors[i].Data = tempLabel.ToArray();
                label_tensors[i].SynchronizeToRemote();
            }
        }

        private float[] CharToVect(char c)
        {
            float[] result = new float[_unique_chars.Count];
            int index = _unique_chars.IndexOf(c);
            result[index] = 1f;
            return result;
        }

        private float[] CharToVect(char[] chars)
        {
            List<float> batch = new List<float>();
            for (int i = 0; i < chars.Length; i++)
            {
                batch.AddRange(CharToVect(chars[i]));
            }
            return batch.ToArray();
        }
    }
}
