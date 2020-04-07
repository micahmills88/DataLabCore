﻿using System;
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
        LossLayer _loss_layer;

        Stopwatch stopwatch;

        List<string> _keys = new List<string>();
        Dictionary<string, LayerConfig> _descriptions = new Dictionary<string, LayerConfig>();
        Dictionary<string, ITrainableLayer> _layers = new Dictionary<string, ITrainableLayer>();

        string _model_name = "";

        public ModelBuilder(string modelName, IDataSource dataSource, ControllerType controllerType)
        {
            _model_name = modelName;
            _data_source = dataSource;
            _controller = new TensorController(controllerType);
        }

        public void AddLayer(LayerConfig layerDescription)
        {
            _keys.Add(layerDescription.Key);
            _descriptions.Add(layerDescription.Key, layerDescription);
        }

        public void AddConvolutionLayer(int filterHeight, int filterWidth, int filterCount, 
            ActivationType activation = ActivationType.ReLU, PaddingType padding = PaddingType.None)
        {
            var desc = new LayerConfig()
            {
                layerType = LayerType.Convolution,
                HasWeights = true,
                WeightRows = filterHeight,
                WeightColumns = filterWidth,
                //WeightLayers
                WeightCubes = filterCount,
                //Weights
                HasBias = true,
                //BiasRows
                //BiasColumns
                BiasLayers = filterCount,
                BiasCubes = 1,
                //Bias
                activationType = activation,
                paddingType = padding
            };

            AddLayer(desc);
        }

        public void AddMaxPoolLayer() //todo add kernel size settings
        {
            var desc = new LayerConfig()
            {
                layerType = LayerType.MaxPool,
                HasWeights = true,
                WeightRows = 2,
                WeightColumns = 2,
            };

            AddLayer(desc);
        }

        public void AddFlattenLayer()
        {
            var desc = new LayerConfig()
            {
                layerType = LayerType.Flatten,
            };

            AddLayer(desc);
        }

        public void AddDenselayer(int outputCount, ActivationType activation = ActivationType.Sigmoid)
        {
            var desc = new LayerConfig()
            {
                layerType = LayerType.Dense,
                HasWeights = true,
                WeightColumns = outputCount,
                WeightLayers = 1,
                WeightCubes = 1,
                HasBias = true,
                BiasRows = 1,
                BiasColumns = outputCount,
                BiasLayers = 1,
                BiasCubes = 1,
                activationType = activation
            };

            AddLayer(desc);
        }

        public void Initialize(int batchSize)
        {
            _data_source.Initialize(_controller, batchSize);
            int inputHeight = _data_source.SampleHeight;
            int inputWidth = _data_source.SampleWidth;
            int inputDepth = _data_source.SampleDepth;
            int inputSize = _data_source.SampleSize;

            for (int i = 0; i < _keys.Count; i++)
            {
                var key = _keys[i];
                var desc = _descriptions[key];
                ITrainableLayer layer = null;
                switch (desc.layerType)
                {
                    case LayerType.Convolution:
                        layer = new ConvolutionLayerBuilder(_controller, inputHeight, inputWidth, inputDepth, batchSize, desc);
                        break;
                    case LayerType.Dense:
                        layer = new DenseLayerBuilder(_controller, inputSize, batchSize, desc);
                        break;
                    case LayerType.Flatten:
                        layer = new FlattenLayerBuilder(_controller, inputHeight, inputWidth, inputDepth, batchSize, desc);
                        break;
                    case LayerType.MaxPool:
                        layer = new MaxPoolLayerBuilder(_controller, inputHeight, inputWidth, inputDepth, batchSize, desc);
                        break;
                }
                _layers.Add(key, layer);
                inputHeight = layer.OutputHeight;
                inputWidth = layer.OutputWidth;
                inputDepth = layer.OutputDepth;
                inputSize = layer.OutputSize;
            }

            var lastLayer = _layers[_keys[_keys.Count - 1]];
            _loss_layer = new LossLayer(_controller, lastLayer.OutputSize, batchSize, _data_source.SampleCount);
        }

        public void TrainModel(float learningRate, LossFunction lossFunction, int epochs, float stopLoss)
        {
            var _reverse_keys = _keys.ToList();
            _reverse_keys.Reverse();

            stopwatch = Stopwatch.StartNew();
            int batchcount = _data_source.GetTotalBatches();
            for (int e = 0; e < epochs; e++)
            {
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

        public ModelConfig ExportModel()
        {
            ModelConfig model = new ModelConfig();
            model.ModelName = _model_name;
            model.LayerConfigs = new List<LayerConfig>();
            foreach(var key in _keys)
            {
                var layer = _layers[key];
                model.LayerConfigs.Add(layer.ExportLayer());
            }
            model.InputRows = _data_source.SampleHeight;
            model.InputColumns = _data_source.SampleWidth;
            model.InputLayers = _data_source.SampleDepth;
            return model;
        }
    }
}