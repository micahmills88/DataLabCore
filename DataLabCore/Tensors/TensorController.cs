using ILGPU;
using ILGPU.Runtime;
using ILGPU.Runtime.Cuda;
using ILGPU.Algorithms;
using ILGPU.Algorithms.ScanReduceOperations;
using System;
using System.Collections.Generic;
using System.Text;

namespace DataLabCore
{
    public class TensorController
    {
        private static Accelerator _accelerator;
        private static TensorKernels _kernels;

        public TensorController()
        {
            var context = new Context();
            context.EnableAlgorithms();
            _accelerator = new CudaAccelerator(context);
            _kernels = new TensorKernels(_accelerator);
        }

        public void Synchronize()
        {
            _accelerator.Synchronize();
        }

        public MemoryBuffer<float> AllocateBuffer(int size)
        {
            return _accelerator.Allocate<float>(size);
        }

        public void DenseForward(Tensor result, Tensor inputs, Tensor weights, Tensor bias, ActivationType activationType)
        {
            _kernels.MatrixMultiply(result.Size, result.DataView, inputs.DataView, weights.DataView, inputs.Columns, weights.Columns);
            _kernels.AddBias(result.Size, result.DataView, bias.DataView);
            if(activationType == ActivationType.Sigmoid)
            {
                _kernels.ActivateSigmoid(result.Size, result.DataView);
            }
            if(activationType == ActivationType.Softmax)
            {
                _kernels.ActivateSoftmax(result.Rows, result.DataView, result.Columns);
            }
        }

        public void ConvolutionForward(Tensor result, Tensor inputs, Tensor weights, Tensor bias, ActivationType activationType)
        {
            _kernels.ForwardCorrelation(result.Size, result.DataView, inputs.DataView, weights.DataView,
                inputs.Rows, inputs.Columns, inputs.Layers, weights.Rows, weights.Columns, weights.Layers, weights.Cubes);
            _kernels.AddBias(result.Size, result.DataView, bias.DataView);
            _kernels.ActivateReLU(result.Size, result.DataView);
        }

        public void DenseOutputError(Tensor outputErrors, Tensor layerOutputs, Tensor errors, ActivationType activationType)
        {
            //transpose copies the data to outputErrors
            _kernels.Transpose2D(outputErrors.Size, outputErrors.DataView, layerOutputs.DataView, layerOutputs.Rows, layerOutputs.Columns);
            outputErrors.Transpose2DValues();
            if(activationType == ActivationType.Sigmoid)
            {
                _kernels.DeriveSigmoid(outputErrors.Size, outputErrors.DataView);
            }
            if(activationType == ActivationType.Softmax)
            {
                _kernels.DeriveSoftmax(outputErrors.Size, outputErrors.DataView);
            }
            _kernels.MultiplyErrors(outputErrors.Size, outputErrors.DataView, errors.DataView);
        }

        public void ConvolutionOutputError(Tensor outputErrors, Tensor layerOutputs, Tensor errors, ActivationType activationType)
        {
            _kernels.DeriveReLU(outputErrors.Size, outputErrors.DataView, layerOutputs.DataView);
            _kernels.MultiplyErrors(outputErrors.Size, outputErrors.DataView, errors.DataView);
        }

        public void DenseInputError(Tensor inputErrors, Tensor weights, Tensor outputErrors)
        {
            _kernels.MatrixMultiply(inputErrors.Size, inputErrors.DataView, weights.DataView, outputErrors.DataView, weights.Columns, outputErrors.Columns);
        }

        public void ConvolutionInputError(Tensor inputErrors, Tensor paddedErrors, Tensor outputErrors, Tensor invertedFilters, Tensor filters)
        {
            var xpad = filters.Columns - 1;
            var ypad = filters.Rows - 1;
            int totalRows = outputErrors.Rows * outputErrors.Layers * outputErrors.Cubes;
            _kernels.Pad(totalRows, paddedErrors.DataView, outputErrors.DataView, outputErrors.Columns, outputErrors.Rows, xpad, ypad);
            int totalFilterLayers = filters.Layers * filters.Cubes;
            _kernels.InvertFilter(totalFilterLayers, invertedFilters.DataView, filters.DataView, filters.LayerSize);
            _kernels.InputErrorConvolution(inputErrors.Size, inputErrors.DataView, paddedErrors.DataView, invertedFilters.DataView,
                paddedErrors.Rows, paddedErrors.Columns, paddedErrors.Layers, invertedFilters.Rows, invertedFilters.Columns, invertedFilters.Layers, invertedFilters.Cubes);
        }

        public void DenseLayerWeightUpdate(Tensor weights, Tensor weightErrors, Tensor weightMomentum, Tensor inputs, Tensor outputErrors, float batchMultiple, float learningRate)
        {
            _kernels.TransposedMatrixMultiply(weightErrors.Size, weightErrors.DataView, inputs.DataView, outputErrors.DataView, outputErrors.Rows, outputErrors.Columns, inputs.Columns);
            _kernels.AdjustMomentum(weightMomentum.Size, weightMomentum.DataView, weightErrors.DataView, batchMultiple);
            _kernels.ApplyGradient(weights.Size, weights.DataView, weightMomentum.DataView, learningRate);
        }

        public void ConvolutionLayerWeightUpdate(Tensor filters, Tensor filterErrors, Tensor filterMomentum, Tensor inputs, Tensor outputErrors, float batchMultiple, float learningRate)
        {
            _kernels.WeightErrorCorrelation(filterErrors.Size, filterErrors.DataView, inputs.DataView, outputErrors.DataView,
                inputs.Rows, inputs.Columns, inputs.Layers, inputs.Cubes, outputErrors.Rows, outputErrors.Columns, outputErrors.Layers);
            _kernels.AdjustMomentum(filterMomentum.Size, filterMomentum.DataView, filterErrors.DataView, batchMultiple);
            _kernels.ApplyGradient(filters.Size, filters.DataView, filterMomentum.DataView, learningRate);
        }

        public void DenseLayerBiasUpdate(Tensor bias, Tensor biasErrors, Tensor biasMomentum, Tensor outputErrors, float batchMultiple, float learningRate)
        {
            _kernels.RowSums(biasErrors.Size, biasErrors.DataView, outputErrors.DataView, outputErrors.Columns);
            _kernels.AdjustMomentum(biasMomentum.Size, biasMomentum.DataView, biasErrors.DataView, batchMultiple);
            _kernels.ApplyGradient(bias.Size, bias.DataView, biasMomentum.DataView, learningRate);
        }

        public void ConvolutionLayerBiasUpdate(Tensor bias, Tensor biasErrors, Tensor biasMomentum, Tensor outputErrors, float batchMultiple, float learningRate)
        {
            _kernels.RowSums(outputErrors.Cubes, biasErrors.DataView, outputErrors.DataView, outputErrors.CubeSize);
            _kernels.AdjustMomentum(biasMomentum.Size, biasMomentum.DataView, biasErrors.DataView, batchMultiple);
            _kernels.ApplyGradient(bias.Size, bias.DataView, biasMomentum.DataView, learningRate);
        }

        public void CalculateLoss(Tensor totalerrors, Tensor errors, Tensor data, Tensor labels, LossFunction lossFunction)
        {
            _kernels.SubtractTransposed(errors.Rows, errors.DataView, data.DataView, labels.DataView, data.Rows, data.Columns);
            //all loss functions sum into the total errors
            if(lossFunction == LossFunction.Logistic)
            {
                _kernels.LogisticLoss(totalerrors.Size, totalerrors.DataView, data.DataView, labels.DataView);
            }
            if(lossFunction == LossFunction.Multiclass)
            {
                _kernels.MultiClassLoss(totalerrors.Size, totalerrors.DataView, data.DataView, labels.DataView);
            }
            if(lossFunction == LossFunction.MeanSquared)
            {
                _kernels.MeanSquaredError(totalerrors.Size, totalerrors.DataView, data.DataView, labels.DataView);
            }
        }
    }
}
