using System;
using System.Collections.Generic;
using System.Text;

namespace DataLabCore
{
    public enum ActivationType
    {
        Sigmoid = 4001,
        Softmax = 4002,
        ReLU = 4003,
        None = 4000
    }

    public enum PaddingType
    {
        None = 1000,
        Same = 1001
    }

    public enum LayerType
    {
        None = 0,
        Convolution = 1,
        Dense = 2,
        Flatten = 3,
        MaxPool = 4
    }

    public class LayerDescription
    {
        public LayerType layerType;
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
        public readonly string Key = Guid.NewGuid().ToString();
    }
}
