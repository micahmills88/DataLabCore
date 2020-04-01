using ILGPU;
using ILGPU.Runtime;
using ILGPU.Runtime.Cuda;
using ILGPU.Runtime.CPU;
using ILGPU.Algorithms;
using ILGPU.Algorithms.ScanReduceOperations;
using System;
using System.Collections.Generic;
using System.Text;

namespace DataLabCore
{
    public enum ControllerType
    {
        CUDA,
        CPU
    }

    public class TensorController
    {
        private static Accelerator _accelerator;
        private static TensorKernels _kernels;

        public TensorController(ControllerType ct)
        {
            var context = new Context();
            context.EnableAlgorithms();
            if(ct == ControllerType.CPU)
            {
                _accelerator = new CPUAccelerator(context);
            }
            if(ct == ControllerType.CUDA)
            {
                _accelerator = new CudaAccelerator(context);
            }
            _kernels = new TensorKernels(_accelerator);
        }

        #region layer methods
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
            if(activationType == ActivationType.ReLU)
            {
                _kernels.ActivateReLU(result.Size, result.DataView);
            }
        }

        public void ConvolutionForward(Tensor result, Tensor inputs, Tensor weights, Tensor bias, ActivationType activationType)
        {
            _kernels.ForwardCorrelation(result.Size, result.DataView, inputs.DataView, weights.DataView,
                inputs.Rows, inputs.Columns, inputs.Layers, weights.Rows, weights.Columns, weights.Layers, weights.Cubes);
            _kernels.AddBias(result.Size, result.DataView, bias.DataView);
            if(activationType == ActivationType.ReLU)
            {
                _kernels.ActivateReLU(result.Size, result.DataView);
            }
            if (activationType == ActivationType.Sigmoid)
            {
                _kernels.ActivateSigmoid(result.Size, result.DataView);
            }
        }

        public void DenseOutputError(Tensor outputErrors, Tensor layerOutputs, Tensor errors, ActivationType activationType)
        {
            //transpose copies the data to outputErrors
            _kernels.Transpose2D(outputErrors.Size, outputErrors.DataView, layerOutputs.DataView, layerOutputs.Rows, layerOutputs.Columns);
            outputErrors.Transpose2DValues();
            if(activationType == ActivationType.Sigmoid)
            {
                _kernels.DeriveSigmoid(outputErrors.Size, outputErrors.DataView, outputErrors.DataView);
            }
            if(activationType == ActivationType.Softmax)
            {
                _kernels.DeriveSoftmax(outputErrors.Size, outputErrors.DataView);
            }
            if(activationType == ActivationType.ReLU)
            {
                _kernels.DeriveReLU(outputErrors.Size, outputErrors.DataView, outputErrors.DataView);
            }
            _kernels.MultiplyErrors(outputErrors.Size, outputErrors.DataView, errors.DataView);
        }

        public void ConvolutionOutputError(Tensor outputErrors, Tensor layerOutputs, Tensor errors, ActivationType activationType)
        {
            if(activationType == ActivationType.ReLU)
            {
                _kernels.DeriveReLU(outputErrors.Size, outputErrors.DataView, layerOutputs.DataView);
            }
            if(activationType == ActivationType.Sigmoid)
            {
                _kernels.DeriveSigmoid(outputErrors.Size, outputErrors.DataView, layerOutputs.DataView);
            }
            _kernels.MultiplyErrors(outputErrors.Size, outputErrors.DataView, errors.DataView);
        }

        public void DenseInputError(Tensor inputErrors, Tensor weights, Tensor outputErrors)
        {
            _kernels.MatrixMultiply(inputErrors.Size, inputErrors.DataView, weights.DataView, outputErrors.DataView, weights.Columns, outputErrors.Columns);
        }

        public void ConvolutionInputError(Tensor inputErrors, Tensor paddedErrors, Tensor outputErrors, Tensor invertedFilters, Tensor filters)
        {
            int xpad = (paddedErrors.Columns - outputErrors.Columns) / 2;
            int ypad = (paddedErrors.Rows - outputErrors.Rows) / 2;
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
            _kernels.SumCubes(biasErrors.Size, biasErrors.DataView, outputErrors.DataView, outputErrors.Cubes, biasErrors.Size);
            _kernels.AdjustMomentum(biasMomentum.Size, biasMomentum.DataView, biasErrors.DataView, batchMultiple);
            _kernels.ApplyGradient(bias.Size, bias.DataView, biasMomentum.DataView, learningRate);
        }

        public void CalculateLoss(Tensor totalerrors, Tensor errors, Tensor data, Tensor labels, LossFunction lossFunction)
        {
            _kernels.SubtractTransposed(errors.Size, errors.DataView, data.DataView, labels.DataView, data.Rows, data.Columns);
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

        public void MaxPoolForward(Tensor inputs, Tensor outputs, Tensor mask)
        {
            _kernels.MaxPoolForward(outputs.Size, mask.DataView, outputs.DataView, inputs.DataView, outputs.Columns, inputs.Columns);
        }

        public void MaxPoolBackward(Tensor inputErrors, Tensor mask, Tensor errors)
        {
            _kernels.MaxPoolBackward(errors.Size, mask.DataView, inputErrors.DataView, errors.DataView, errors.Columns, inputErrors.Columns);
        }
        #endregion layer methods

        #region raw methods
        public void MatrixMultiply(Tensor result, Tensor left, Tensor right)
        {
            _kernels.MatrixMultiply(result.Size, result.DataView, left.DataView, right.DataView, left.Columns, right.Columns);
        }

        public void AddBias(Tensor result, Tensor bias)
        {
            _kernels.AddBias(result.Size, result.DataView, bias.DataView);
        }

        public void Transpose2D(Tensor result, Tensor input, bool setValues = true)
        {
            _kernels.Transpose2D(result.Size, result.DataView, input.DataView, input.Rows, input.Columns);
            if(setValues)
            {
                result.Transpose2DValues();
            }
        }

        public void TransposedMatrixMultiply(Tensor outputs, Tensor left, Tensor right)
        {
            _kernels.TransposedMatrixMultiply(outputs.Size, outputs.DataView, left.DataView, right.DataView, outputs.Columns, right.Columns, left.Columns);
        }

        public void RowSum(Tensor result, Tensor values)
        {
            _kernels.RowSums(result.Size, result.DataView, values.DataView, values.Columns);
        }

        public void SubtractTransposed(Tensor errors, Tensor data, Tensor labels)
        {
            _kernels.SubtractTransposed(errors.Size, errors.DataView, data.DataView, labels.DataView, data.Rows, data.Columns);
        }


        public void ForwardCorrelation(Tensor output, Tensor input, Tensor filters)
        {
            _kernels.ForwardCorrelation(output.Size, output.DataView, input.DataView, filters.DataView,
                input.Rows, input.Columns, input.Layers, filters.Rows, filters.Columns, filters.Layers, filters.Cubes);
        }

        public void BackwardConvolution(Tensor result, Tensor errors, Tensor filters)
        {
            _kernels.InputErrorConvolution(result.Size, result.DataView, errors.DataView, filters.DataView,
                errors.Rows, errors.Columns, errors.Layers, filters.Rows, filters.Columns, filters.Layers, filters.Cubes);
        }

        public void WeightErrorCorrelation(Tensor result, Tensor inputs, Tensor errors)
        {
            _kernels.WeightErrorCorrelation(result.Size, result.DataView, inputs.DataView, errors.DataView,
                inputs.Rows, inputs.Columns, inputs.Layers, inputs.Cubes, errors.Rows, errors.Columns, errors.Layers);
        }

        public void PadTensor(Tensor result, Tensor inputs, int xpad, int ypad)
        {
            int totalRows = inputs.Rows * inputs.Layers * inputs.Cubes;
            _kernels.Pad(totalRows, result.DataView, inputs.DataView, inputs.Columns, inputs.Rows, xpad, ypad);
        }

        public void InvertFilters(Tensor result, Tensor filters)
        {
            int totalFilterLayers = filters.Layers * filters.Cubes;
            _kernels.InvertFilter(totalFilterLayers, result.DataView, filters.DataView, filters.LayerSize);
        }

        public void SumCubes(Tensor output, Tensor input)
        {
            _kernels.SumCubes(output.Size, output.DataView, input.DataView, input.Cubes, output.Size);
        }

        public void SoftMax(Tensor outputs)
        {
            _kernels.ActivateSoftmax(outputs.Rows, outputs.DataView, outputs.Columns);
        }
        #endregion raw methods
    }
}
