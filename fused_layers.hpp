#pragma once

#include "layers.hpp"
/*
 * @brief Class representing the fused Conv -> BatchNorm -> Activ (Relu) layer
 * @details [long description]
 * 
 * @param input_dims [description]
 * @param channels_out [description]
 * @param kernel_size [description]
 * @param padding [description]
 * @param stride [description]
 * @param bn_mode [description]
 * @param g [description]
 * @param e [description]
 * @param e [description]
 * @return [description]
 */
struct FusedCNRLayer: public ConvDesc, public ConvLayerDesc, public Layer {
    miopenConvFwdAlgorithm_t fwd_algo;
    Tensor weights;
    std::shared_ptr<Tensor> input_ptr;
    // Hold a ref to the fusion plan and associated data
    miopenActivationDescriptor_t desc;


    miopenFusionOpDescriptor_t bNormOp;
    miopenFusionOpDescriptor_t convoOp;
    miopenFusionOpDescriptor_t activOp;

    miopenBatchNormMode_t bn_mode = miopenBNSpatial;
    TensorDesc bn_dim;
    miopenTensorDescriptor_t biasScaleTensor;
    Tensor scale;
    Tensor bias;
    double exp;
    Tensor running_mean;
    Tensor running_var;
    double epsilon;

    miopenFusionPlanDescriptor_t fusePlanDesc;
    miopenOperatorArgs_t fusionArgs;

    static TensorDesc get_bn_dim(const TensorDesc& input_dim, miopenBatchNormMode_t bn_mode) {
        TensorDesc bn(0,0,0,0);
        CHECK_MIO(miopenDeriveBNTensorDescriptor(bn.desc, input_dim.desc, bn_mode));
        bn.update_get();
        return bn;
    }

    FusedCNRLayer(const TensorDesc& input_dims, int channels_out, int kernel_size, int padding, int stride,
                  miopenBatchNormMode_t bn_mode_=miopenBNSpatial, double eps = 1e-05):
                  ConvDesc(padding, padding, stride, stride, 1, 1),
                  ConvLayerDesc({input_dims.n, input_dims.h, input_dims.w, input_dims.c, channels_out, kernel_size, padding, stride}),
                  Layer((Dim&)input_dims, getConvOutputDim(padding, stride, input_dims, TensorDesc(channels_out, input_dims.c, kernel_size, kernel_size))),
                  weights(channels_out, input_dims.c, kernel_size, kernel_size),
                  bn_mode(bn_mode_),bn_dim(get_bn_dim(input_dim, bn_mode)), scale(bn_dim), bias(bn_dim), 
                  running_mean(bn_dim), running_var(bn_dim), epsilon(eps)

    {
      // create the fusion plan and 
       miopenCreateFusionPlan(&fusePlanDesc, miopenVerticalFusion, input_desc.desc);
       miopenCreateOperatorArgs(&fusionArgs);
    }

    void init_forward(const Tensor& input, Tensor&* output) override
    {
        // hardcode the algorithm to the supported one
        CHECK_MIO(miopenCreateOpConvForwardAlgo(fusePlanDesc,
                                  &convoOp,
                                  this->desc,
                                  miopenConvolutionFwdAlgoDirect,
                                  weights.desc));

        CHECK_MIO(miopenDeriveBNTensorDescriptor(biasScaleTensor, input.desc, bn_mode));
        CHECK_MIO(miopenCreateOpBatchNormInference(fusePlanDesc, &bNormOp, bn_mode, biasScaleTensor));

        // we are only concerned with RELU
        CHECK_MIO(miopenCreateOpActivationForward(fusePlanDesc, &activOp, miopenActivationRELU))

        // compile fusion plan
        CHECK_MIO(miopenCompileFusionPlan(mio::handle(), fusePlanDesc));

        //Set the Args
        float alpha = static_cast<float>(1), beta = static_cast<float>(0);
        miopenSetOpArgsConvForward(fusionArgs, convoOp, &alpha, &beta, weights.data);

        float activ_alpha = static_cast<float>(0), activ_beta = static_cast<float>(0), activ_gamma = static_cast<float>(0);

        miopenSetOpArgsActivForward(
                fusionArgs, activOp, &alpha, &beta, activ_alpha, activ_beta, activ_gamma);

        miopenSetOpArgsBatchNormInference(fusionArgs,
                                  bNormOp,
                                  &alpha,
                                  &beta,
                                  scale.data,
                                  bias.data,
                                  running_mean.data,
                                  running_var.data,
                                  epsilon);


    }

    void forward(const Tensor& input, Tensor& output) override
    {
        // Execute the fusion plan 
        miopenExecuteFusionPlan(mio::handle(),
                                fusePlanDesc,
                                input.desc,
                                input.data,
                                output.desc,
                                output.data,
                                fusionArgs);

    }
}