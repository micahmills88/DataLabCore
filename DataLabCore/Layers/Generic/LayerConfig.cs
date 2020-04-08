using System;
using System.Collections.Generic;
using System.Text;

namespace DataLabCore
{
    public enum ActivationType
    {
        None = 3000,
        Sigmoid = 3001,
        Softmax = 3002,
        ReLU = 3003        
    }

    public enum PaddingType
    {
        None = 2000,
        Same = 2001
    }

    public enum LayerType
    {
        None = 1000,
        Convolution = 1001,
        Dense = 1002,
        Flatten = 1003,
        MaxPool = 1004
    }

    public class LayerConfig
    {
        public string Id = "";
        public LayerType layerType = LayerType.None;
        public int LayerIndex = 0;
        public bool HasWeights = false;
        public int WeightRows = 0;
        public int WeightColumns = 0;
        public int WeightLayers = 0;
        public int WeightCubes = 0;
        public float[] Weights;
        public bool HasBias = false;
        public int BiasRows = 0;
        public int BiasColumns = 0;
        public int BiasLayers = 0;
        public int BiasCubes = 0;
        public float[] Bias;
        public ActivationType activationType = ActivationType.None;
        public PaddingType paddingType = PaddingType.None;

        public LayerDescription ToLayerDescription()
        {
            var desc = new LayerDescription();
            desc.Id = this.Id;
            desc.layerType = this.layerType;
            desc.HasWeights = this.HasWeights;
            desc.WeightRows = this.WeightRows;
            desc.WeightColumns = this.WeightColumns;
            desc.WeightLayers = this.WeightLayers;
            desc.WeightCubes = this.WeightCubes;
            desc.HasBias = this.HasBias;
            desc.BiasRows = this.BiasRows;
            desc.BiasColumns = this.BiasColumns;
            desc.BiasLayers = this.BiasLayers;
            desc.BiasCubes = this.BiasCubes;
            desc.activationType = this.activationType;
            desc.paddingType = this.paddingType;
            return desc;
        }
    }
}
