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
    public class ModelTrainer
    {
        TensorController _controller;
        IDataSource _data_source;
        LossLayer _loss_layer;


        List<string> _keys = new List<string>();
        Dictionary<string, LayerDescription> _descriptions = new Dictionary<string, LayerDescription>();
        Dictionary<string, ITrainableLayer> _layers = new Dictionary<string, ITrainableLayer>();

        public ModelTrainer(IDataSource dataSource, ControllerType controllerType)
        {
            _data_source = dataSource;
            _controller = new TensorController(controllerType);
        }

        public void AddLayer(LayerDescription layerDescription)
        {
            _keys.Add(layerDescription.Key);
            _descriptions.Add(layerDescription.Key, layerDescription);
        }

        public void AddConvolutionLayer(int filterHeight, int filterWidth, int filterCount, 
            ActivationType activation = ActivationType.ReLU, PaddingType padding = PaddingType.None)
        {
            var desc = new LayerDescription()
            {
                layerType = LayerType.Convolution,
                HasWeights = true,
                WeightRows = filterHeight,
                WeightColumns = filterWidth,
                WeightCubes = filterCount,
                HasBias = true,
                BiasLayers = filterCount,
                BiasCubes = 1,
                activationType = activation,
                paddingType = padding
            };

            AddLayer(desc);
        }

        public void AddMaxPoolLayer() //todo add kernel size settings
        {
            var desc = new LayerDescription()
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
            var desc = new LayerDescription()
            {
                layerType = LayerType.Flatten,
            };

            AddLayer(desc);
        }

        public void AddDenselayer(int outputCount, ActivationType activation = ActivationType.Sigmoid)
        {
            var desc = new LayerDescription()
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
            for (int i = 0; i < _keys.Count; i++)
            {
                //initialize each layertrainer
            }

            //var lastLayer = get last
            _loss_layer = new LossLayer(_controller, 0, batchSize, _data_source.SampleCount)
        }

        public void TrainModel(float learningRate, LossFunction lossFunction)
        {
            var _reverse_keys = _keys.ToList();
            _reverse_keys.Reverse();

            var data = _data_source.GetSampleBatch(0);
            var labels = _data_source.GetLabelBatch(0);
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
    }
}
