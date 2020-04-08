using MongoDB.Bson;
using MongoDB.Bson.Serialization.Attributes;
using System;
using System.Collections.Generic;
using System.Text;

namespace DataLabCore
{
    public class LayerDescription
    {
        [BsonId]
        [BsonRepresentation(BsonType.ObjectId)]
        public string Id;
        [BsonRepresentation(BsonType.Int32)]
        public LayerType layerType = LayerType.None;
        [BsonRepresentation(BsonType.Int32)]
        public ActivationType activationType = ActivationType.None;
        [BsonRepresentation(BsonType.Int32)]
        public PaddingType paddingType = PaddingType.None;

        public int LayerIndex;

        public bool HasWeights = false;
        public int WeightRows = 0;
        public int WeightColumns = 0;
        public int WeightLayers = 0;
        public int WeightCubes = 0;
        
        public bool HasBias = false;
        public int BiasRows = 0;
        public int BiasColumns = 0;
        public int BiasLayers = 0;
        public int BiasCubes = 0;

        [BsonRepresentation(BsonType.ObjectId)]
        public string WeightFileID;
        [BsonRepresentation(BsonType.ObjectId)]
        public string BiasFileID;

        [BsonRepresentation(BsonType.ObjectId)]
        public string ModelId;

        public LayerConfig ToNewLayerConfig()
        {
            var config = new LayerConfig();
            config.Id = this.Id;
            config.LayerIndex = this.LayerIndex;
            return config;
        }

        public LayerConfig ToLayerConfig()
        {
            var cfg = new LayerConfig();
            cfg.Id = this.Id;
            cfg.layerType = this.layerType;
            cfg.LayerIndex = this.LayerIndex;
            cfg.HasWeights = this.HasWeights;
            cfg.WeightRows = this.WeightRows;
            cfg.WeightColumns = this.WeightColumns;
            cfg.WeightLayers = this.WeightLayers;
            cfg.WeightCubes = this.WeightCubes;
            cfg.HasBias = this.HasBias;
            cfg.BiasRows = this.BiasRows;
            cfg.BiasColumns = this.BiasColumns;
            cfg.BiasLayers = this.BiasLayers;
            cfg.BiasCubes = this.BiasCubes;
            cfg.activationType = this.activationType;
            cfg.paddingType = this.paddingType;
            return cfg;
        }
    }
}
