using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace DataLabCore
{

    public class ModelRunner
    {
        public TensorController _controller;
        ModelDatabase _model_db;
        ModelConfig _model_config;

        List<string> _keys = new List<string>();
        Dictionary<string, IProductionLayer> _layers = new Dictionary<string, IProductionLayer>();

        int _batch_size;

        public static ModelRunner LoadFromDatabase(string databaseURI, string modelID, ControllerType controllerType, int inHeight, int inWidth, int inDepth, int batchSize)
        {
            var db = new ModelDatabase(databaseURI);
            var cfg = db.LoadModel(modelID);
            var mr = new ModelRunner(cfg, batchSize, controllerType);
            mr.LoadLayers(inHeight, inWidth, inDepth);
            return mr;
        }

        private ModelRunner(ModelConfig config, int batchSize, ControllerType ct)
        {
            _model_config = config;
            _batch_size = batchSize;
            _controller = new TensorController(ct);
        }

        private void LoadLayers(int inHeight, int inWidth, int inDepth)
        {
            var configs = _model_config.LayerConfigs.Values.OrderBy(item => item.LayerIndex).ToList();
            var lay = InitializeLayer(inHeight, inWidth, inDepth, inHeight*inWidth*inDepth, configs[0]);
            for (int i = 1; i < configs.Count; i++)
            {
                lay = InitializeLayer(lay.OutputHeight, lay.OutputWidth, lay.OutputDepth, lay.OutputSize, configs[i]);
            }
        }

        private IProductionLayer InitializeLayer(int height, int width, int depth, int size, LayerConfig config)
        {
            IProductionLayer layer = null;
            switch (config.layerType)
            {
                case LayerType.Convolution:
                    layer = new ConvolutionLayer(_controller, height, width, depth, _batch_size, config);
                    break;
                case LayerType.Dense:
                    layer = new DenseLayer(_controller, size, _batch_size, config);
                    break;
                case LayerType.Flatten:
                    layer = new FlattenLayer(_controller, height, width, depth, _batch_size, config);
                    break;
                case LayerType.MaxPool:
                    layer = new MaxPoolLayer(_controller, height, width, depth, _batch_size, config);
                    break;
            }
            _layers.Add(config.Id, layer);
            _keys.Add(config.Id);
            return layer;
        }

        public Tensor Process(Tensor sample)
        {
            Tensor result = _layers[_keys[0]].Forward(sample);
            for (int i = 1; i < _keys.Count; i++)
            {
                result = _layers[_keys[i]].Forward(result);
            }
            return result;
        }
    }
}
