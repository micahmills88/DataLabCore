using MongoDB.Bson;
using MongoDB.Bson.Serialization.Attributes;
using System;
using System.Collections.Generic;
using System.Text;

namespace DataLabCore
{
    public class ModelDescription
    {
        [BsonId]
        [BsonRepresentation(BsonType.ObjectId)]
        public string Id;
        public string Name;

        public ModelConfig ToModelConfig()
        {
            var config = new ModelConfig();
            config.Id = this.Id;
            config.ModelName = this.Name;
            config.LayerConfigs = new Dictionary<string, LayerConfig>();
            return config;
        }
    }
}
