using System;
using System.IO;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

using DataLabCore;

using OSGeo.GDAL;
using OSGeo.OGR;
using OSGeo.OSR;

namespace DataLabCore
{
    public class DataSource_GEO : IDataSource
    {
        TensorController _controller;
        int _width = 2;
        int _height = 1;
        int _sample_size = 2;
        int _depth = 1;
        int _classes = 2;
        int _samplecount = 1000000;
        int _batchsize = 10;

        public int SampleHeight { get => _height; }
        public int SampleWidth { get => _width; }
        public int SampleDepth { get => _depth; }
        public int SampleSize { get => _sample_size; }
        public int SampleCount { get => _samplecount; }
        public int Classes { get => _classes; }

        List<string> keys = new List<string>();
        Dictionary<string, float[]> data_labels = new Dictionary<string, float[]>();
        Dictionary<string, float[]> data_samples = new Dictionary<string, float[]>();

        List<Tensor> sample_tensors = new List<Tensor>();
        List<Tensor> label_tensors = new List<Tensor>();

        double[] transform = new double[6];

        public DataSource_GEO()
        {
            var random = new Random();
            Gdal.AllRegister();
            var raster = Gdal.Open(@"M:\maps\data\DEM_Mosaic_WV_Statewide_1-to-3-m_UTM17_p2021.tif", Access.GA_ReadOnly);
            string projection = raster.GetProjection();
            raster.GetGeoTransform(transform);

            var x_min = transform[0];
            var x_max = x_min + raster.RasterXSize;
            var y_min = transform[3] - raster.RasterYSize;
            var y_max = transform[3];

            var sr_wgs84 = new SpatialReference("");
            sr_wgs84.ImportFromEPSG(4326);
            var sr_raster = new SpatialReference(projection);
            var ct = new CoordinateTransformation(sr_raster, sr_wgs84);

            for (int i = 0; i < _samplecount; i++)
            {
                //get random nad83
                var xpercent = random.NextDouble();
                var ypercent = random.NextDouble();
                var randx = (xpercent * raster.RasterXSize) + x_min;
                var randy = (ypercent * raster.RasterYSize) + y_min;

                //get lat long
                var p = new double[3];
                ct.TransformPoint(p, randx, randy, 0);
                //p[0] is lat p[1] is lon

                float[] sample = new float[] { (float)p[0] / 10f, (float)p[1] / 10f };
                float[] label = new float[] 
                { 
                    (float)xpercent,
                    (float)ypercent
                };

                string guid = Guid.NewGuid().ToString("N");
                keys.Add(guid);

                data_samples.Add(guid, sample);
                data_labels.Add(guid, label);
            }
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
                sample_tensors.Add(new Tensor(_controller, _height, _width, _depth, _batchsize, new float[s_size]));
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
