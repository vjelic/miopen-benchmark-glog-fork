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

#if 0
struct FusedCNRLayer: public ConvDesc, public ConvLayerDesc, public Layer {
    Tensor weights;
    // Hold a ref to the fusion plan and associated data
    // miopenActivationDescriptor_t desc;


    miopenFusionOpDescriptor_t bNormOp;
    miopenFusionOpDescriptor_t convoOp;
    miopenFusionOpDescriptor_t activOp;

    miopenBatchNormMode_t bn_mode = miopenBNSpatial;
    TensorDesc bn_dim;
    Tensor scale;
    Tensor bias;
    double exp;
    Tensor running_mean;
    Tensor running_var;
    double epsilon;

    std::shared_ptr<Tensor> conv_output = nullptr;
    miopenConvFwdAlgorithm_t fwd_algo;

    int mode = 0; // 0: CNR fused, 1: Conv separate, NR fused
    miopenFusionPlanDescriptor_t fusePlanDesc;
    miopenOperatorArgs_t fusionArgs;

    virtual std::ostream& write_name(std::ostream& os) const 
    {
        //return os << "Conv(" << kernel_size << "x" << kernel_size << ")";
        return os << "FusedCNRLayer(" << kernel_size << "x" << kernel_size << ",pad=" << padding << ",s=" << stride << ",m=" << mode << ")";
    }

    static TensorDesc get_bn_dim(const TensorDesc& input_dim, miopenBatchNormMode_t bn_mode) {
        TensorDesc bn(0,0,0,0);
        CHECK_MIO(miopenDeriveBNTensorDescriptor(bn.desc, input_dim.desc, bn_mode));
        bn.update_get();
        return bn;
    }

    FusedCNRLayer(const TensorDesc& input_dims, int channels_out, int kernel_size, int padding=0, int stride=1,
                  miopenBatchNormMode_t bn_mode_=miopenBNSpatial, double eps = 1e-05):
                  ConvDesc(padding, padding, stride, stride, 1, 1),
                  ConvLayerDesc({input_dims.n, input_dims.h, input_dims.w, input_dims.c, channels_out, kernel_size, padding, stride}),
                  Layer((Dim&)input_dims, getConvOutputDim(padding, stride, input_dims, TensorDesc(channels_out, input_dims.c, kernel_size, kernel_size))),
                  weights(channels_out, input_dims.c, kernel_size, kernel_size),
                  bn_mode(bn_mode_),bn_dim(get_bn_dim(input_dims, bn_mode)), scale(bn_dim), bias(bn_dim), 
                  running_mean(bn_dim), running_var(bn_dim), epsilon(eps)
    {
      // create the fusion plan and 
       miopenCreateFusionPlan(&fusePlanDesc, miopenVerticalFusion, input_desc.desc);
       miopenCreateOperatorArgs(&fusionArgs);
    }

    void init_forward(const Tensor& input, Tensor& output) override
    {
        float alpha = static_cast<float>(1), beta = static_cast<float>(0);
        // hardcode the algorithm to the supported one
        CHECK_MIO(miopenCreateOpConvForwardAlgo(fusePlanDesc,
                                  &convoOp,
                                  this->desc,
                                  miopenConvolutionFwdAlgoDirect,
                                  weights.desc));

        CHECK_MIO(miopenDeriveBNTensorDescriptor(bias.desc, input.desc, bn_mode));
        bias.update_get();
        CHECK_MIO(miopenCreateOpBatchNormInference(fusePlanDesc, &bNormOp, bn_mode, bias.desc));

        // we are only concerned with RELU
        CHECK_MIO(miopenCreateOpActivationForward(fusePlanDesc, &activOp, miopenActivationRELU))

        // compile fusion plan
        auto status = miopenCompileFusionPlan(mio::handle(), fusePlanDesc);
        if(status == miopenStatusSuccess)
        {
            mode = 0;
            //Set the Args
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
        else
        {
          mode = 1;
          miopenDestroyFusionPlanDescriptor(fusePlanDesc);

          int n, c, h, w;
          CHECK_MIO(miopenGetConvolutionForwardOutputDim(this->desc, input.desc, weights.desc, 
                    &n, &c, &h, &w));

          conv_output = std::make_shared<Tensor>(n, c, h, w);

          miopenCreateFusionPlan(&fusePlanDesc, miopenVerticalFusion, conv_output->desc);
          bn_dim = get_bn_dim(*conv_output,bn_mode);
          bias = Tensor(bn_dim);
          scale = Tensor(bn_dim);
          running_mean = Tensor(bn_dim);
          running_var = Tensor(bn_dim);
          //CHECK_MIO(miopenDeriveBNTensorDescriptor(bias.desc, conv_output->desc, bn_mode));
          //bias.update_get();
          CHECK_MIO(miopenCreateOpBatchNormInference(fusePlanDesc, &bNormOp, bn_mode, bias.desc));
          CHECK_MIO(miopenCreateOpActivationForward(fusePlanDesc, &activOp, miopenActivationRELU));
          CHECK_MIO(miopenCompileFusionPlan(mio::handle(), fusePlanDesc));

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

          // setup the discrete conv op
          size_t fwd_workspace_size;
          CHECK_MIO(miopenConvolutionForwardGetWorkSpaceSize(mio::handle(), weights.desc, input.desc, this->desc, output.desc, &fwd_workspace_size));
          DEBUG("Init fwd " << *this << " req workspace: " << fwd_workspace_size);

          DevBuffer& buffer = WorkSpace::get(fwd_workspace_size);

          // find best algo, and benchmark!
          miopenConvAlgoPerf_t perfs[4];
          int returned_algos;
          CHECK_MIO(miopenFindConvolutionForwardAlgorithm(mio::handle(), input.desc, input.data, 
            weights.desc, weights.data, this->desc, output.desc, output.data, 4, 
            &returned_algos, perfs, buffer.data, fwd_workspace_size, false));

          INFO("\tMIOpen Found " << returned_algos << " fwd algorithms, choosing " << perfs[0].fwd_algo << ": ");
          for (int i = 0; i < returned_algos; ++i) {
              INFO("\t\t" << i << ") " << perfs[i].fwd_algo << " - time: " << perfs[i].time << ", Memory: " << perfs[i].memory);
          }

          fwd_algo = perfs[0].fwd_algo;
        }
    }

    void forward(const Tensor& input, Tensor& output) override
    {
      if(mode == 1)
      {
        // Launch the convolution
        float alpha = 1.f;
        float beta = 0.f;
        DevBuffer& buffer = WorkSpace::get();
        CHECK_MIO(miopenConvolutionForward(mio::handle(), &alpha, input.desc, input.data, 
          weights.desc, weights.data, this->desc, fwd_algo, &beta, conv_output->desc, 
          conv_output->data, buffer.data, buffer.size));

        CHECK_MIO(miopenExecuteFusionPlan(mio::handle(), fusePlanDesc, conv_output->desc, conv_output->data,
                                output.desc, output.data, fusionArgs));
      }
      else
      {
        // Execute the fusion plan 
        CHECK_MIO(miopenExecuteFusionPlan(mio::handle(),
                                fusePlanDesc,
                                input.desc,
                                input.data,
                                output.desc,
                                output.data,
                                fusionArgs));
      }
    }

    void backward(const Tensor& doutput, Tensor& dinput)
    {
      assert(false);
    }
};
#endif

struct FusedCBR: public ConvDesc, public ConvLayerDesc, public Layer {
    Tensor weights;
    miopenFusionOpDescriptor_t biasOp;
    miopenFusionOpDescriptor_t convoOp;
    miopenFusionOpDescriptor_t activOp;

    Tensor bias;

    miopenConvFwdAlgorithm_t fwd_algo;

    // unfused equivalent
    std::shared_ptr<Sequential> model = nullptr;


    int mode = 0; // 0: CNR fused, 1: Conv separate, NR fused
    miopenFusionPlanDescriptor_t fusePlanDesc;
    miopenOperatorArgs_t fusionArgs;
    bool is_fused_faster = true;

    virtual std::ostream& write_name(std::ostream& os) const 
    {
        //return os << "Conv(" << kernel_size << "x" << kernel_size << ")";
        return os << "FusedCBR(" << kernel_size << "x" << kernel_size << ",pad=" << padding << ",s=" << stride << ",is_fused_faster=" << is_fused_faster << ")";
    }

    FusedCBR(const TensorDesc& input_dims, int channels_out, int kernel_size, int padding=0, int stride=1):
                  ConvDesc(padding, padding, stride, stride, 1, 1),
                  ConvLayerDesc({input_dims.n, input_dims.h, input_dims.w, input_dims.c, channels_out, kernel_size, padding, stride}),
                  Layer((Dim&)input_dims, getConvOutputDim(padding, stride, input_dims, TensorDesc(channels_out, input_dims.c, kernel_size, kernel_size))),
                  weights(channels_out, input_dims.c, kernel_size, kernel_size), bias(1, channels_out,1 ,1)
    {
      // create the fusion plan and 
       miopenCreateFusionPlan(&fusePlanDesc, miopenVerticalFusion, input_desc.desc);
       miopenCreateOperatorArgs(&fusionArgs);

       model = std::make_shared<Sequential>(input_dims, "FusedCBR_discrete");
       model->emplace<ConvLayer>(channels_out, kernel_size, padding, stride );
       model->emplace<BatchNormInference>();
       model->emplace<ReLU>();
    }

    void init_forward(const Tensor& input, Tensor& output) override
    {
        float alpha = static_cast<float>(1), beta = static_cast<float>(0);
        // hardcode the algorithm to the supported one
        CHECK_MIO(miopenCreateOpConvForwardAlgo(fusePlanDesc,
                                  &convoOp,
                                  this->desc,
                                  miopenConvolutionFwdAlgoDirect,
                                  weights.desc));


        CHECK_MIO(miopenCreateOpBiasForward(fusePlanDesc, &biasOp, bias.desc));
        // we are only concerned with RELU
        CHECK_MIO(miopenCreateOpActivationForward(fusePlanDesc, &activOp, miopenActivationRELU))

        // compile fusion plan
        auto status = miopenCompileFusionPlan(mio::handle(), fusePlanDesc);
        if(status == miopenStatusSuccess)
        {
            mode = 0;
            //Set the Args
            miopenSetOpArgsConvForward(fusionArgs, convoOp, &alpha, &beta, weights.data);

            float activ_alpha = static_cast<float>(0), activ_beta = static_cast<float>(0), activ_gamma = static_cast<float>(0);

            miopenSetOpArgsActivForward(
                    fusionArgs, activOp, &alpha, &beta, activ_alpha, activ_beta, activ_gamma);
            miopenSetOpArgsBiasForward(fusionArgs, biasOp, &alpha, &beta, bias.data);
        }
        else
        {
          mode = 1;
          assert(false);
        }
        auto iters = 20;
        is_fused_faster = true;
        //Initialize the discrete model
        model->init_forward(input, output);
        //dummy run to warm up the cache
        model->forward(input, output);
        this->forward(input, output);

        // Run the benchmark on each and time 
        auto disc_start = std::chrono::steady_clock::now();
        for(auto idx = 0 ; idx < iters; idx++)
        {
          model->forward(input, output);
        }
        auto disc_end = std::chrono::steady_clock::now();
        auto disc_time = ((disc_end - disc_start) / iters);

        auto fused_start = std::chrono::steady_clock::now();
        for(auto idx = 0; idx < iters; idx++)
        {
          this->forward(input, output);
        }
        auto fused_end = std::chrono::steady_clock::now();
        auto fused_time = ((fused_end - fused_start) / iters);

        // Free up the memory from the slower model
        if(fused_time < disc_time)
        {
          is_fused_faster = true;
          model.reset();
        }
        else
        {
          is_fused_faster = false;
          weights.dealloc();
          bias.dealloc();
        }
    }

    void forward(const Tensor& input, Tensor& output) override
    {

      if(is_fused_faster)
      {
        // Execute the fusion plan 
        CHECK_MIO(miopenExecuteFusionPlan(mio::handle(),
                                fusePlanDesc,
                                input.desc,
                                input.data,
                                output.desc,
                                output.data,
                                fusionArgs));
      }
      else
      {
        model->forward(input, output);
      }
    }

    void backward(const Tensor& doutput, Tensor& dinput)
    {
      assert(false);
    }
};

struct FusedCB: public ConvDesc, public ConvLayerDesc, public Layer {
    Tensor weights;

    miopenFusionOpDescriptor_t biasOp;
    miopenFusionOpDescriptor_t convoOp;

    Tensor bias;

    miopenConvFwdAlgorithm_t fwd_algo;

    int mode = 0; // 0: CNR fused, 1: Conv separate, NR fused
    miopenFusionPlanDescriptor_t fusePlanDesc;
    miopenOperatorArgs_t fusionArgs;

    std::shared_ptr<Sequential> model = nullptr;
    bool is_fused_faster = true;


    virtual std::ostream& write_name(std::ostream& os) const 
    {
        //return os << "Conv(" << kernel_size << "x" << kernel_size << ")";
        return os << "FusedCB(" << kernel_size << "x" << kernel_size << ",pad=" << padding << ",s=" << stride << ",is_fused_faster=" << is_fused_faster << ")";
    }

    FusedCB(const TensorDesc& input_dims, int channels_out, int kernel_size, int padding=0, int stride=1):
                  ConvDesc(padding, padding, stride, stride, 1, 1),
                  ConvLayerDesc({input_dims.n, input_dims.h, input_dims.w, input_dims.c, channels_out, kernel_size, padding, stride}),
                  Layer((Dim&)input_dims, getConvOutputDim(padding, stride, input_dims, TensorDesc(channels_out, input_dims.c, kernel_size, kernel_size))),
                  weights(channels_out, input_dims.c, kernel_size, kernel_size), bias(1, channels_out, 1, 1)
    {
      // create the fusion plan and 
       miopenCreateFusionPlan(&fusePlanDesc, miopenVerticalFusion, input_desc.desc);
       miopenCreateOperatorArgs(&fusionArgs);

       model = std::make_shared<Sequential>(input_dims, "FusedCB_discrete");
       model->emplace<ConvLayer>(channels_out, kernel_size, padding, stride );
       model->emplace<BatchNormInference>();
    }

    void init_forward(const Tensor& input, Tensor& output) override
    {
        float alpha = static_cast<float>(1), beta = static_cast<float>(0);
        // hardcode the algorithm to the supported one
        CHECK_MIO(miopenCreateOpConvForwardAlgo(fusePlanDesc,
                                  &convoOp,
                                  this->desc,
                                  miopenConvolutionFwdAlgoDirect,
                                  weights.desc));

        CHECK_MIO(miopenCreateOpBiasForward(fusePlanDesc, &biasOp, bias.desc));
        // compile fusion plan
        auto status = miopenCompileFusionPlan(mio::handle(), fusePlanDesc);
        if(status == miopenStatusSuccess)
        {
            mode = 0;
            //Set the Args
            miopenSetOpArgsConvForward(fusionArgs, convoOp, &alpha, &beta, weights.data);
            miopenSetOpArgsBiasForward(fusionArgs, biasOp, &alpha, &beta, bias.data);
        }
        else
        {
          mode = 1;
          assert(false);
        }

        auto iters = 20;
        is_fused_faster = true;
        //Initialize the discrete model
        model->init_forward(input, output);
        //dummy run to warm up the cache
        model->forward(input, output);
        this->forward(input, output);

        // Run the benchmark on each and time 
        auto disc_start = std::chrono::steady_clock::now();
        for(auto idx = 0 ; idx < iters; idx++)
        {
          model->forward(input, output);
        }
        auto disc_end = std::chrono::steady_clock::now();
        auto disc_time = ((disc_end - disc_start) / iters);

        auto fused_start = std::chrono::steady_clock::now();
        for(auto idx = 0; idx < iters; idx++)
        {
          this->forward(input, output);
        }
        auto fused_end = std::chrono::steady_clock::now();
        auto fused_time = ((fused_end - fused_start) / iters);

        // Free up the memory from the slower model
        if(fused_time < disc_time)
        {
          is_fused_faster = true;
          model.reset();
        }
        else
        {
          is_fused_faster = false;
          weights.dealloc();
          bias.dealloc();
        }
    }

    void forward(const Tensor& input, Tensor& output) override
    {

      if(is_fused_faster)
      {
        // Execute the fusion plan 
        CHECK_MIO(miopenExecuteFusionPlan(mio::handle(),
                                fusePlanDesc,
                                input.desc,
                                input.data,
                                output.desc,
                                output.data,
                                fusionArgs));
      }
      else
      {
        model->forward(input, output);
      }
    }

    void backward(const Tensor& doutput, Tensor& dinput)
    {
      assert(false);
    }
};

struct FusedBNR: public Layer {

    miopenFusionOpDescriptor_t bNormOp;
    miopenFusionOpDescriptor_t activOp;

    miopenBatchNormMode_t bn_mode = miopenBNSpatial;
    TensorDesc bn_dim;
    Tensor scale;
    Tensor bias;
    double exp;
    Tensor running_mean;
    Tensor running_var;
    double epsilon;

    int mode = 0; // 0: NR fused, 1:Not fused
    miopenFusionPlanDescriptor_t fusePlanDesc;
    miopenOperatorArgs_t fusionArgs;

    std::shared_ptr<Sequential> model = nullptr; 
    bool is_fused_faster = true;

    virtual std::ostream& write_name(std::ostream& os) const 
    {
        //return os << "Conv(" << kernel_size << "x" << kernel_size << ")";
        return os << "FusedBNR(" << "is_fused_faster=" << is_fused_faster << ")";
    }

    static TensorDesc get_bn_dim(const TensorDesc& input_dim, miopenBatchNormMode_t bn_mode) {
        TensorDesc bn(0,0,0,0);
        CHECK_MIO(miopenDeriveBNTensorDescriptor(bn.desc, input_dim.desc, bn_mode));
        bn.update_get();
        return bn;
    }

    FusedBNR(const TensorDesc& input_dims, miopenBatchNormMode_t bn_mode_=miopenBNSpatial, double eps = 1e-05):
                  Layer(input_dims, input_dims),
                  bn_mode(bn_mode_),bn_dim(get_bn_dim(input_dims, bn_mode)), scale(bn_dim), bias(bn_dim), 
                  running_mean(bn_dim), running_var(bn_dim), epsilon(eps)
    {
      // create the fusion plan and 
       miopenCreateFusionPlan(&fusePlanDesc, miopenVerticalFusion, input_desc.desc);
       miopenCreateOperatorArgs(&fusionArgs);

       model = std::make_shared<Sequential>(input_dims, "FusedCBR_discrete");
       model->emplace<BatchNormInference>();
       model->emplace<ReLU>();
    }

    void init_forward(const Tensor& input, Tensor& output) override
    {
        float alpha = static_cast<float>(1), beta = static_cast<float>(0);
        CHECK_MIO(miopenCreateOpBatchNormInference(fusePlanDesc, &bNormOp, bn_mode, bias.desc));

        // we are only concerned with RELU
        CHECK_MIO(miopenCreateOpActivationForward(fusePlanDesc, &activOp, miopenActivationRELU))

        // compile fusion plan
        auto status = miopenCompileFusionPlan(mio::handle(), fusePlanDesc);
        if(status == miopenStatusSuccess)
        {
            mode = 0;
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
        else
        {
          mode = 1;
          assert(false);
        }

        auto iters = 20;
        is_fused_faster = true;
        //Initialize the discrete model
        model->init_forward(input, output);
        //dummy run to warm up the cache
        model->forward(input, output);
        this->forward(input, output);

        // Run the benchmark on each and time 
        auto disc_start = std::chrono::steady_clock::now();
        for(auto idx = 0 ; idx < iters; idx++)
        {
          model->forward(input, output);
        }
        auto disc_end = std::chrono::steady_clock::now();
        auto disc_time = ((disc_end - disc_start) / iters);

        auto fused_start = std::chrono::steady_clock::now();
        for(auto idx = 0; idx < iters; idx++)
        {
          this->forward(input, output);
        }
        auto fused_end = std::chrono::steady_clock::now();
        auto fused_time = ((fused_end - fused_start) / iters);

        // Free up the memory from the slower model
        if(fused_time < disc_time)
        {
          is_fused_faster = true;
          model.reset();
        }
        else
        {
          is_fused_faster = false;
          scale.dealloc();
          bias.dealloc();
          running_mean.dealloc();
          running_var.dealloc();
        }
    }

    void forward(const Tensor& input, Tensor& output) override
    {
      if(is_fused_faster)
      {
        // Execute the fusion plan 
        CHECK_MIO(miopenExecuteFusionPlan(mio::handle(),
                                fusePlanDesc,
                                input.desc,
                                input.data,
                                output.desc,
                                output.data,
                                fusionArgs));
      }
      else
      {
        model->forward(input, output);
      }
    }

    void backward(const Tensor& doutput, Tensor& dinput)
    {
      assert(false);
    }
};
