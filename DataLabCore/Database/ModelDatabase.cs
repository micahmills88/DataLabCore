using MongoDB.Driver;
using MongoDB.Driver.GridFS;
using System;
using System.Collections.Generic;
using System.Text;
using System.Runtime.InteropServices;
using System.Xml.Linq;
using System.ComponentModel;
using MongoDB.Bson;

namespace DataLabCore
{
    public class ModelDatabase
    {
        MongoClient _client;
        GridFSBucket _gridfs;
        IMongoCollection<LayerDescription> _layers;
        IMongoCollection<ModelDescription> _models;

        public ModelDatabase(string uri)
        {
            _client = new MongoClient(uri);
            var database = _client.GetDatabase("DataLabCore");
            _gridfs = new GridFSBucket(database);
            _models = database.GetCollection<ModelDescription>("ModelDescriptions");
            _layers = database.GetCollection<LayerDescription>("LayerDescriptions");
        }

        public ModelConfig GetNewModelConfig(string name)
        {
            var desc = new ModelDescription()
            {
                Name = name
            };

            _models.InsertOne(desc);
            return desc.ToModelConfig();
        }

        public LayerConfig GetNewLayerConfig(string modelId, int layerIndex)
        {
            var desc = new LayerDescription();
            desc.ModelId = modelId;
            desc.LayerIndex = layerIndex;
            _layers.InsertOne(desc);
            return desc.ToNewLayerConfig();
        }

        public void SaveModelLayers(ModelConfig modelConfig)
        {
            foreach(var key in modelConfig.LayerConfigs.Keys)
            {
                var cfg = modelConfig.LayerConfigs[key];
                var weightId = cfg.HasWeights ? SaveLayerDataArray(cfg.Weights, "weights_" + cfg.Id) : null;
                var biasId = cfg.HasBias ? SaveLayerDataArray(cfg.Bias, "bias_" + cfg.Id) : null;
                var desc = cfg.ToLayerDescription();
                desc.WeightFileID = weightId;
                desc.BiasFileID = biasId;
                desc.ModelId = modelConfig.Id;
                _layers.FindOneAndReplace(
                    Builders<LayerDescription>.Filter.Eq(item => item.Id, cfg.Id),
                    desc
                );
            }
        }

        private string SaveLayerDataArray(float[] weights, string filename)
        {
            Span<float> floats = new Span<float>(weights);
            var bytes = MemoryMarshal.Cast<float, byte>(floats).ToArray();
            var id = _gridfs.UploadFromBytes(filename, bytes);
            return id.ToString();
        }

        private float[] LoadLayerDataArray(string id)
        {
            var bytes = _gridfs.DownloadAsBytes(new ObjectId(id));
            Span<byte> src = new Span<byte>(bytes);
            var floats = MemoryMarshal.Cast<byte, float>(src).ToArray();
            return floats;
        }

        public ModelConfig LoadModel(string id)
        {
            var modelDesc = _models.Find(item => item.Id == id).FirstOrDefault();
            if (modelDesc == null) return null;
            var modelConfig = modelDesc.ToModelConfig();
            var layers = _layers.Find(item => item.ModelId == id).ToList();
            foreach (var layer in layers)
            {
                var laycfg = layer.ToLayerConfig();
                if (layer.HasWeights)
                {
                    laycfg.Weights = LoadLayerDataArray(layer.WeightFileID);
                }
                if (layer.HasBias)
                {
                    laycfg.Bias = LoadLayerDataArray(layer.BiasFileID);
                }
                modelConfig.LayerConfigs.Add(laycfg.Id, laycfg);
            }
            return modelConfig;
        }
    }
}
