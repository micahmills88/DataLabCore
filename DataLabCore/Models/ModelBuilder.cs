using System;
using System.Collections.Generic;
using System.Text;
using System.Linq;

using ILGPU;
using ILGPU.Runtime;
using ILGPU.Runtime.Cuda;
using System.Diagnostics;

namespace DataLabCore
{
    public class ModelBuilder
    {
        TensorController _controller;
        IDataSource _data_source;
        int _batch_size = 10;
        LossLayer _loss_layer;

        ModelDatabase _model_db;
        ModelConfig _model_config;
        Stopwatch stopwatch;

        List<string> _keys = new List<string>();
        Dictionary<string, ITrainableLayer> _layers = new Dictionary<string, ITrainableLayer>();

        public static ModelBuilder CreateNewModel(string databaseURI, string modelName, int batchSize, IDataSource dataSource, ControllerType controllerType)
        {
            var db = new ModelDatabase(databaseURI);
            var cfg = db.GetNewModelConfig(modelName);
            return new ModelBuilder(db, cfg, batchSize, dataSource, controllerType);
        }

        public static ModelBuilder LoadFromDatabase(string databaseURI, string modelID, int batchSize, IDataSource dataSource, ControllerType controllerType)
        {
            var db = new ModelDatabase(databaseURI);
            var cfg = db.LoadModel(modelID);
            var mb =  new ModelBuilder(db, cfg, batchSize, dataSource, controllerType);
            mb.LoadLayers();
            return mb;
        }

        private ModelBuilder(ModelDatabase db, ModelConfig cfg, int batchSize, IDataSource dataSource, ControllerType controllerType)
        {
            _model_db = db;
            _model_config = cfg;
            _batch_size = batchSize;
            _data_source = dataSource;
            _controller = new TensorController(controllerType);
            _data_source.Initialize(_controller, batchSize);
        }

        public void AddNewLayer(LayerConfig layerConfig)
        {
            _model_config.LayerConfigs.Add(layerConfig.Id, layerConfig);

            if (layerConfig.LayerIndex == 0)
            {
                InitializeLayer(_data_source.SampleHeight, _data_source.SampleWidth, _data_source.SampleDepth, _data_source.SampleSize, layerConfig, true);
            }
            else
            {
                var lastLayer = _layers[_keys.Last()];
                InitializeLayer(lastLayer.OutputHeight, lastLayer.OutputWidth, lastLayer.OutputDepth, lastLayer.OutputSize, layerConfig, true);
            }
        }

        private void LoadLayers()
        {
            var configs = _model_config.LayerConfigs.Values.OrderBy(item => item.LayerIndex).ToList();
            var lay = InitializeLayer(_data_source.SampleHeight, _data_source.SampleWidth, _data_source.SampleDepth, _data_source.SampleSize, configs[0], false);
            for (int i = 1; i < configs.Count; i++)
            {
                lay = InitializeLayer(lay.OutputHeight, lay.OutputWidth, lay.OutputDepth, lay.OutputSize, configs[i], false);
            }
        }

        private ITrainableLayer InitializeLayer(int height, int width, int depth, int size, LayerConfig config, bool newLayer)
        {
            ITrainableLayer layer = null;
            switch (config.layerType)
            {
                case LayerType.Convolution:
                    layer = new ConvolutionLayerBuilder(_controller, height, width, depth, _batch_size, config, newLayer);
                    break;
                case LayerType.Dense:
                    layer = new DenseLayerBuilder(_controller, size, _batch_size, config, newLayer);
                    break;
                case LayerType.Flatten:
                    layer = new FlattenLayerBuilder(_controller, height, width, depth, _batch_size, config);
                    break;
                case LayerType.MaxPool:
                    layer = new MaxPoolLayerBuilder(_controller, height, width, depth, _batch_size, config);
                    break;
            }
            _layers.Add(config.Id, layer);
            _keys.Add(config.Id);
            return layer;
        }

        #region addlayers
        public void AddConvolutionLayer(int filterHeight, int filterWidth, int filterCount, 
            ActivationType activation = ActivationType.ReLU, PaddingType padding = PaddingType.None)
        {
            var cfg = _model_db.GetNewLayerConfig(_model_config.Id, _model_config.LayerConfigs.Count);
            cfg.layerType = LayerType.Convolution;
            cfg.HasWeights = true;
            cfg.WeightRows = filterHeight;
            cfg.WeightColumns = filterWidth;
            //WeightLayers
            cfg.WeightCubes = filterCount;
            //Weights
            cfg.HasBias = true;
            //BiasRows
            //BiasColumns
            cfg.BiasLayers = filterCount;
            cfg.BiasCubes = 1;
            //Bias
            cfg.activationType = activation;
            cfg.paddingType = padding;

            AddNewLayer(cfg);
        }

        public void AddMaxPoolLayer() //todo add kernel size settings
        {
            var cfg = _model_db.GetNewLayerConfig(_model_config.Id, _model_config.LayerConfigs.Count);
            cfg.layerType = LayerType.MaxPool;
            cfg.HasWeights = true;
            cfg.WeightRows = 2;
            cfg.WeightColumns = 2;

            AddNewLayer(cfg);
        }

        public void AddFlattenLayer()
        {
            var cfg = _model_db.GetNewLayerConfig(_model_config.Id, _model_config.LayerConfigs.Count);
            cfg.layerType = LayerType.Flatten;

            AddNewLayer(cfg);
        }

        public void AddDenselayer(int outputCount, ActivationType activation = ActivationType.Sigmoid)
        {
            var cfg = _model_db.GetNewLayerConfig(_model_config.Id, _model_config.LayerConfigs.Count);
            cfg.layerType = LayerType.Dense;
            cfg.HasWeights = true;
            cfg.WeightColumns = outputCount;
            cfg.WeightLayers = 1;
            cfg.WeightCubes = 1;
            cfg.HasBias = true;
            cfg.BiasRows = 1;
            cfg.BiasColumns = outputCount;
            cfg.BiasLayers = 1;
            cfg.BiasCubes = 1;
            cfg.activationType = activation;

            AddNewLayer(cfg);
        }
        #endregion

        public void TrainModel(float learningRate, LossFunction lossFunction, int epochs, float stopLoss)
        {
            var _reverse_keys = _keys.ToList();
            _reverse_keys.Reverse();

            if(_loss_layer == null)
            {
                var lastlayer = _layers[_reverse_keys[0]];
                _loss_layer = new LossLayer(_controller, lastlayer.OutputSize, _batch_size, _data_source.SampleCount);
            }

            stopwatch = Stopwatch.StartNew();
            int batchcount = _data_source.GetTotalBatches();
            for (int e = 0; e < epochs; e++)
            {
                stopwatch.Restart();
                for (int b = 0; b < batchcount; b++)
                {
                    var data = _data_source.GetSampleBatch(b);
                    var labels = _data_source.GetLabelBatch(b);
                    for (int i = 0; i < _keys.Count; i++)
                    {
                        data = _layers[_keys[i]].Forward(data);
                    }
                    var error = _loss_layer.CalculateLoss(data, labels, lossFunction);
                    for (int i = 0; i < _reverse_keys.Count; i++)
                    {
                        error = _layers[_reverse_keys[i]].Backward(error, learningRate, i < _reverse_keys.Count - 1);
                    }
                }

                stopwatch.Stop();
                var epochLoss = _loss_layer.GetEpochLoss();
                Console.WriteLine("Epoch {0:D4} Time {1} Loss {2:N5}", e, stopwatch.ElapsedMilliseconds, epochLoss);

                _data_source.Shuffle();
            }
        }

        public void SaveModelToDatabase()
        {
            foreach(var layer in _layers.Values)
            {
                layer.ExportLayer();
            }
            _model_db.SaveModelLayers(_model_config);
        }
    }
}
