using System;
using System.Collections.Generic;
using System.Text;

namespace DataLabCore
{
    public class ModelConfig
    {
        public string ModelName = "";
        public String ModelID;
        public List<LayerConfig> LayerConfigs;
        public int InputRows = 0;
        public int InputColumns = 0;
        public int InputLayers = 0;
    }
}
