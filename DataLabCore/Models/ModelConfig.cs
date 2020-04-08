using System;
using System.Collections.Generic;
using System.Text;

namespace DataLabCore
{
    public class ModelConfig
    {
        public string ModelName = "";
        public String Id;
        public Dictionary<string, LayerConfig> LayerConfigs;

        public ModelDescription ToModelDescription()
        {
            var desc = new ModelDescription();
            desc.Id = this.Id;
            desc.Name = this.ModelName;
            return desc;
        }
    }
}
