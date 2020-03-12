using System;
using System.Collections.Generic;
using System.IO;
using System.Text;

namespace DataLabCore
{
    public class DataSource
    {
        bool flatten = true;
        int _width = 28;
        int _height = 28;
        int _depth = 1;
        int _classes = 10;
        int _samplecount = 60000;
        int _batchsize = 10;

        string samplePath = @"F:\Machine_Learning\Datasets\MNIST\train-images.idx3-ubyte";
        string labelPath = @"F:\Machine_Learning\Datasets\MNIST\train-lables.idx1-ubyte";

        List<string> keys = new List<string>();
        Dictionary<string, float[]> data_labels = new Dictionary<string, float[]>();
        Dictionary<string, float[]> data_samples = new Dictionary<string, float[]>();

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
        }

        public Tensor GetSampleBatch(int batchnum, int batchsize)
        {
            List<float> values = new List<float>();
            int start = batchnum * batchsize;
            foreach (var key in keys.GetRange(start, batchsize))
            {
                values.AddRange(data_samples[key]);
            }

            if (flatten)
            {
                return new Tensor(batchsize, _height * _width * _depth, values.ToArray());
            }
            return new Tensor(_height, _width, _depth, batchsize, values.ToArray());
        }

        public Tensor GetLabelBatch(int batchnum, int batchsize)
        {
            List<float> values = new List<float>();
            int start = batchnum * batchsize;
            foreach (var key in keys.GetRange(start, batchsize))
            {
                values.AddRange(data_labels[key]);
            }
            return new Tensor(batchsize, _classes, values.ToArray());
        }

        public void Shuffle()
        {
            keys.Shuffle();
        }
    }
}
