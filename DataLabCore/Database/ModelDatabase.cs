using MongoDB.Driver;
using MongoDB.Driver.GridFS;
using System;
using System.Collections.Generic;
using System.Text;

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
            var database = _client.GetDatabase("DataSets");
            _gridfs = new GridFSBucket(database);
            _models = database.GetCollection<ModelDescription>("ModelDescriptions");
            _layers = database.GetCollection<LayerDescription>("LayerDescriptions");
        }

        public void SaveModel(ModelConfig modelConfig, LayerConfig[] layerConfigs)
        {

        }

        public void SaveModelDescription(ModelDescription modelDescription)
        {

        }

        public void SaveLayerDescription(LayerDescription layerDescription)
        {

        }
    }
}
