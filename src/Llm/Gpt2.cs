using System;
using System.Diagnostics;
using System.IO;
using System.Runtime.InteropServices;

namespace nietras.LargeLanguageModel;

#pragma warning disable IDE0007 // Use implicit type

internal static partial class Gpt2
{
    static readonly Action<string> Log = t => { Console.WriteLine(t); Trace.WriteLine(t); };
    static readonly Action<string> LogNoNewLine = t => { Console.Write(t); Trace.Write(t); };

    // Wrap implementation of LLM methods in timing capable
    static TimeLlm CreateTimeLlm(ILlm llm) => new(llm);
    // Skip detailed timing for initial steps
    const int JitAndWarmupCount = 3;

    // ----------------------------------------------------------------------------
    // GPT-2 model definition
    // ----------------------------------------------------------------------------
    public unsafe class Model
    {
        public Model(Config config)
        {
            Config = config;
            Parameters = ParameterTensors.Create(config);
            ParameterGradients = null; // ParameterTensorsNew.Create(config);
        }

        public Config Config { get; }

        // weights (parameters) of the model, and their sizes
        public ParameterTensors Parameters { get; }
        // gradients of the weights (parameters)
        public ParameterTensors? ParameterGradients { get; set; }
        // buffers for the AdamW optimizer
        public float* m_memory;
        public float* v_memory;

        // activations of the model, and their sizes
        public OutputTensors Outputs;
        public nint[] OutputSizes = new nint[OutputTensorCount];
        public nint OutputCount;
        // gradients of the outputs
        public OutputTensors OutputGradients;

        // other run state configuration
        public int Batchsize; // the batch size (B) of current forward pass
        public int TokenCount; // the sequence length (T) of current forward pass
    }

    public record Config
    {
        public int MaxTokenCount; // max sequence length, e.g. 1024
        public int VocabularySize; // vocab size, e.g. 50257
        public int LayerCount; // number of layers, e.g. 12
        public int HeadCount; // number of heads in attention, e.g. 12
        public int ChannelCount; // number of channels, e.g. 768
    }

    unsafe interface ITensorPtrs { public float* MemoryPtr { get; } }

    public sealed class ParameterTensors(in Config c, object s) : Tensors<float>(s)
    {
        public static ParameterTensors Create(in Config c)
        {
            var s = new State();
            var tensors = new ParameterTensors(c, s);
            s.Ntv = new Ntv<float>(s.TotalCount);
            return tensors;
        }

        // Implicitly depends on property initialization following declared
        // order of properties.

        public Tensor<float> TokenEmbeddings { get; } = New([c.VocabularySize, c.ChannelCount], s);
        public Tensor<float> PositionEmbeddings { get; } = New([c.MaxTokenCount, c.ChannelCount], s);

        public Tensor<float> LayerNorm1Weights { get; } = New([c.LayerCount, c.ChannelCount], s);
        public Tensor<float> LayerNorm1Bias { get; } = New([c.LayerCount, c.ChannelCount], s);

        public Tensor<float> QKVWeights { get; } = New([c.LayerCount, 3 * c.ChannelCount, c.ChannelCount], s);
        public Tensor<float> QKVBias { get; } = New([c.LayerCount, 3 * c.ChannelCount], s);

        public Tensor<float> AttentionProjectionWeights { get; } = New([c.LayerCount, c.ChannelCount, c.ChannelCount], s);
        public Tensor<float> AttentionProjectionBias { get; } = New([c.LayerCount, c.ChannelCount], s);

        public Tensor<float> LayerNorm2Weights { get; } = New([c.LayerCount, c.ChannelCount], s);
        public Tensor<float> LayerNorm2Bias { get; } = New([c.LayerCount, c.ChannelCount], s);

        public Tensor<float> FullConnectWeights { get; } = New([c.LayerCount, 4 * c.ChannelCount, c.ChannelCount], s);
        public Tensor<float> FullConnectBias { get; } = New([c.LayerCount, 4 * c.ChannelCount], s);

        public Tensor<float> FullConnectProjectionWeights { get; } = New([c.LayerCount, c.ChannelCount, 4 * c.ChannelCount], s);
        public Tensor<float> FullConnectProjectionBias { get; } = New([c.LayerCount, c.ChannelCount], s);

        public Tensor<float> LayerNormFinalWeights { get; } = New([c.ChannelCount], s);
        public Tensor<float> LayerNormFinalBias { get; } = New([c.ChannelCount], s);
    }

    const int OutputTensorCount = 23;
    [StructLayout(LayoutKind.Sequential)]
    public unsafe struct OutputTensors : ITensorPtrs
    {
        public float* MemoryPtr => Embeded;

        public float* Embeded; // (B, T, C)
        public float* LayerNorm1; // (L, B, T, C)
        public float* LayerNorm1Mean; // (L, B, T)
        public float* LayerNorm1InvStdDev; // (L, B, T)
        public float* QueryKeyValue; // (L, B, T, 3*C)
        // float* preAttention, float* postAttention, float* output
        public float* Attention; // (L, B, T, C)
        public float* PreAttention; // (L, B, NH, T, T)
        public float* PostAttention; // (L, B, NH, T, T)
        public float* AttentionProjected; // (L, B, T, C)
        public float* Residual2; // (L, B, T, C)
        public float* LayerNorm2; // (L, B, T, C)
        public float* LayerNorm2Mean; // (L, B, T)
        public float* LayerNorm2InvStdDev; // (L, B, T)
        public float* FullyConnected; // (L, B, T, 4*C)
        public float* FullyConnectedGeLU; // (L, B, T, 4*C)
        public float* FullyConnectedProjected; // (L, B, T, C)
        public float* Residual3; // (L, B, T, C)
        public float* LayerNormFinal; // (B, T, C)
        public float* LayerNormFinaMean; // (B, T)
        public float* LayerNormInvStdDev; // (B, T)
        public float* Logits; // (B, T, V)
        public float* Probabilities; // (B, T, V)
        public float* Losses; // (B, T)
    }

    public unsafe static Model BuildFromCheckpoint(string checkpointFilePath)
    {
        // read in model from a checkpoint file
        using var file = File.OpenRead(checkpointFilePath);
        Span<int> header = stackalloc int[256];
        // read span from model_file
        file.ReadExactlyUnmanaged(header);
        //fread(model_header, sizeof(int), 256, model_file);
        if (header[0] != 20240326) { throw new InvalidDataException($"Bad magic model file"); }
        if (header[1] != 1) { throw new InvalidDataException($"Bad version in model file"); }

        // read in hyperparameters
        int maxT, V, L, NH, C;
        var config = new Config()
        {
            MaxTokenCount = maxT = header[2],
            VocabularySize = V = header[3],
            LayerCount = L = header[4],
            HeadCount = NH = header[5],
            ChannelCount = C = header[6],
        };
        Log("[GPT-2]");
        Log($"MaxTokenCount: {maxT}");
        Log($"VocabularySize: {V}");
        Log($"LayerCount: {L}");
        Log($"HeadCount: {NH}");
        Log($"ChannelCount: {C}");

        var model = new Model(config);
        // read in all the parameters from file
        file.ReadExactlyUnmanaged(model.Parameters.MemoryPtr, model.Parameters.TotalCount);
        Log($"ParameterCount: {model.Parameters.TotalCount}");

        // other inits
        model.Outputs = default;
        model.m_memory = null;
        model.v_memory = null;
        model.Batchsize = 0;
        model.TokenCount = 0;

        return model;
    }

    internal readonly record struct TrainStepTimings(double Total_ms, double Forward_ms, double ZeroGrad_ms, double Backward_ms, double Update_ms);
    internal readonly record struct TrainStepResult(float Loss, TrainStepTimings Timings);

    static readonly double s_tickstoMs = 1000.0 / Stopwatch.Frequency;

    internal static unsafe string ToReport(this TrainStepTimings t)
    {
        return $"{t.Total_ms,5:F0} ms = Forward {t.Forward_ms,5:F0} ms ZeroGrad {t.ZeroGrad_ms,3:F0} ms Backward {t.Backward_ms,4:F0} ms Update {t.Update_ms,4:F0} ms";
    }

    internal static unsafe TrainStepResult TrainStep(Model model,
        int* inputTokenIndices, int* targetTokenIndices, int batchSize, int tokenCount,
        TimeLlm llm, int step)
    {
        var t0 = Stopwatch.GetTimestamp();
        var loss = Forward(model, inputTokenIndices, targetTokenIndices, batchSize, tokenCount, llm);
        var t1 = Stopwatch.GetTimestamp();
        ZeroGrad(model, llm);
        var t2 = Stopwatch.GetTimestamp();
        Backward(model, inputTokenIndices, targetTokenIndices, llm);
        var t3 = Stopwatch.GetTimestamp();
        Update(model, learningRate: 1e-4f, beta1: 0.9f, beta2: 0.999f,
               eps: 1e-8f, weightDecay: 0.01f, step + 1, llm);
        var t4 = Stopwatch.GetTimestamp();
        TrainStepTimings timings = new((t4 - t0) * s_tickstoMs,
            (t1 - t0) * s_tickstoMs, (t2 - t1) * s_tickstoMs, (t3 - t2) * s_tickstoMs, (t4 - t3) * s_tickstoMs);
        return new(loss, timings);
    }

    static unsafe float Forward(Model model, int* inputs, int* targetTokenIndices, int B, int T, TimeLlm llm)
    {
        // targetTokenIndices are optional and could be null

        // ensure the model was initialized or error output
        if (model.Parameters.MemoryPtr == null)
        {
            throw new InvalidOperationException("Error: model was not initialized properly.");
        }

        // convenience parameters
        int V = model.Config.VocabularySize;
        int L = model.Config.LayerCount;
        int NH = model.Config.HeadCount;
        int C = model.Config.ChannelCount;

        EnsureOutputMemory(model, B, T, V, L, NH, C);

        llm.Part = "0." + nameof(Forward);
        llm.Index = -1;

        // forward pass
        var parameters = model.Parameters; // for brevity
        ref var outputs = ref model.Outputs;
        llm.EmbedForward(inputs, parameters.TokenEmbeddings, parameters.PositionEmbeddings, B, T, C, outputs.Embeded); // encoding goes into residual[0]
        var layersStartIndex = llm.Index;
        for (int l = 0; l < L; l++)
        {
            llm.Index = layersStartIndex;
            var residual = l == 0 ? outputs.Embeded : outputs.Residual3 + (l - 1) * B * T * C;

            // get the pointers of the weights for this layer
            float* l_ln1w = parameters.LayerNorm1Weights.StrideToPtrAt(l);
            float* l_ln1b = parameters.LayerNorm1Bias.StrideToPtrAt(l);
            float* l_qkvw = parameters.QKVWeights.StrideToPtrAt(l);
            float* l_qkvb = parameters.QKVBias.StrideToPtrAt(l);
            float* l_attprojw = parameters.AttentionProjectionWeights.StrideToPtrAt(l);
            float* l_attprojb = parameters.AttentionProjectionBias.StrideToPtrAt(l);
            float* l_ln2w = parameters.LayerNorm2Weights.StrideToPtrAt(l);
            float* l_ln2b = parameters.LayerNorm2Bias.StrideToPtrAt(l);
            float* l_fcw = parameters.FullConnectWeights.StrideToPtrAt(l);
            float* l_fcb = parameters.FullConnectBias.StrideToPtrAt(l);
            float* l_fcprojw = parameters.FullConnectProjectionWeights.StrideToPtrAt(l);
            float* l_fcprojb = parameters.FullConnectProjectionBias.StrideToPtrAt(l);

            // get the pointers of the activations for this layer
            float* l_ln1 = outputs.LayerNorm1 + l * B * T * C;
            float* l_ln1_mean = outputs.LayerNorm1Mean + l * B * T;
            float* l_ln1_rstd = outputs.LayerNorm1InvStdDev + l * B * T;
            float* l_qkv = outputs.QueryKeyValue + l * B * T * 3 * C;
            float* l_atty = outputs.Attention + l * B * T * C;
            float* l_preatt = outputs.PreAttention + l * B * NH * T * T;
            float* l_att = outputs.PostAttention + l * B * NH * T * T;
            float* l_attproj = outputs.AttentionProjected + l * B * T * C;
            float* l_residual2 = outputs.Residual2 + l * B * T * C;
            float* l_ln2 = outputs.LayerNorm2 + l * B * T * C;
            float* l_ln2_mean = outputs.LayerNorm2Mean + l * B * T;
            float* l_ln2_rstd = outputs.LayerNorm2InvStdDev + l * B * T;
            float* l_fch = outputs.FullyConnected + l * B * T * 4 * C;
            float* l_fch_gelu = outputs.FullyConnectedGeLU + l * B * T * 4 * C;
            float* l_fcproj = outputs.FullyConnectedProjected + l * B * T * C;
            float* l_residual3 = outputs.Residual3 + l * B * T * C;

            // now do the forward pass
            llm.LayerNormForward(residual, l_ln1w, l_ln1b, B, T, C, l_ln1_mean, l_ln1_rstd, l_ln1);
            llm.MatMulForward(l_ln1, l_qkvw, l_qkvb, B, T, C, 3 * C, l_qkv);
            llm.AttentionForward(l_qkv, B, T, C, NH, l_preatt, l_att, l_atty);
            llm.MatMulForward(l_atty, l_attprojw, l_attprojb, B, T, C, C, l_attproj);
            llm.ResidualForward(residual, l_attproj, B * T * C, l_residual2);
            llm.LayerNormForward(l_residual2, l_ln2w, l_ln2b, B, T, C, l_ln2_mean, l_ln2_rstd, l_ln2);
            llm.MatMulForward(l_ln2, l_fcw, l_fcb, B, T, C, 4 * C, l_fch);
            llm.GeLUForward(l_fch, B * T * 4 * C, l_fch_gelu);
            llm.MatMulForward(l_fch_gelu, l_fcprojw, l_fcprojb, B, T, 4 * C, C, l_fcproj);
            llm.ResidualForward(l_residual2, l_fcproj, B * T * C, l_residual3);
        }
        var lastResidual = outputs.Residual3 + (L - 1) * B * T * C; // last residual is in residual3
        llm.LayerNormForward(lastResidual, parameters.LayerNormFinalWeights, parameters.LayerNormFinalBias, B, T, C, outputs.LayerNormFinaMean, outputs.LayerNormInvStdDev, outputs.LayerNormFinal);
        llm.MatMulForward(outputs.LayerNormFinal, parameters.TokenEmbeddings, null, B, T, C, V, outputs.Logits);
        llm.SoftmaxForward(outputs.Logits, B, T, V, outputs.Probabilities);

        // also forward the cross-entropy loss function if we have the targetTokenIndices
        if (targetTokenIndices != null)
        {
            llm.CrossEntropyForward(model.Outputs.Probabilities, targetTokenIndices, B, T, V, model.Outputs.Losses);
            // for convenience also evaluate the mean loss
            float meanLoss = 0.0f;
            for (int i = 0; i < B * T; i++) { meanLoss += model.Outputs.Losses[i]; }
            meanLoss /= B * T;
            return meanLoss;
        }
        else
        {
            // if we don't have targetTokenIndices, we don't have a loss
            return -1;
        }

    }

    static unsafe void ZeroGrad(Model model, TimeLlm llm)
    {
        llm.Part = "1." + nameof(ZeroGrad);
        llm.Index = -1;
        if (model.ParameterGradients is not null) { llm.Zero(model.ParameterGradients.MemoryPtr, model.Parameters.TotalCount); }
        if (model.OutputGradients.MemoryPtr != null) { llm.Zero(model.OutputGradients.MemoryPtr, model.OutputCount); }
    }

    static unsafe void Backward(Model model, int* inputTokenIndices, int* targetTokenIndices, TimeLlm llm)
    {
        // lazily allocate the memory for gradients of the weights and activations, if needed
        if (model.ParameterGradients is null)
        {
            model.ParameterGradients ??= ParameterTensors.Create(model.Config);
            model.OutputGradients = AllocateAndSetPointers<OutputTensors>(model.OutputSizes);
            ZeroGrad(model, llm);
        }

        // convenience shortcuts
        int B = model.Batchsize;
        int T = model.TokenCount;
        int V = model.Config.VocabularySize;
        int L = model.Config.LayerCount;
        int NH = model.Config.HeadCount;
        int C = model.Config.ChannelCount;

        // backward pass: go in the reverse order of the forward pass, and call backward() functions
        var parameters = model.Parameters; // for brevity
        var grads = model.ParameterGradients;
        OutputTensors acts = model.Outputs;
        OutputTensors grads_acts = model.OutputGradients;

        // we kick off the chain rule by filling in dlosses with 1.0f/(B*T)
        // technically this is a small, inline backward() pass of calculating
        // total, final loss as the mean over all losses over all (B,T) positions in the batch
        float dloss_mean = 1.0f / (B * T);
        for (int i = 0; i < B * T; i++) { grads_acts.Losses[i] = dloss_mean; }

        llm.Part = "2." + nameof(Backward);
        llm.Index = -1;

        llm.CrossEntropySoftmaxBackward(grads_acts.Losses, acts.Probabilities, targetTokenIndices, B, T, V, grads_acts.Logits);
        llm.MatMulBackward(grads_acts.Logits, acts.LayerNormFinal, parameters.TokenEmbeddings, B, T, C, V, grads.TokenEmbeddings, null, grads_acts.LayerNormFinal);
        float* residual = acts.Residual3 + (L - 1) * B * T * C; // last layer's residual
        float* dresidual = grads_acts.Residual3 + (L - 1) * B * T * C; // write to last layer's residual
        llm.LayerNormBackward(grads_acts.LayerNormFinal, residual, parameters.LayerNormFinalWeights, acts.LayerNormFinaMean, acts.LayerNormInvStdDev, B, T, C, grads.LayerNormFinalWeights, grads.LayerNormFinalBias, dresidual);

        var layerStartIndex = llm.Index;
        for (int l = L - 1; l >= 0; l--)
        {
            llm.Index = layerStartIndex;

            residual = l == 0 ? acts.Embeded : acts.Residual3 + (l - 1) * B * T * C;
            dresidual = l == 0 ? grads_acts.Embeded : grads_acts.Residual3 + (l - 1) * B * T * C;

            // get the pointers of the weights for this layer
            float* l_ln1w = parameters.LayerNorm1Weights.StrideToPtrAt(l);
            float* l_qkvw = parameters.QKVWeights.StrideToPtrAt(l);
            float* l_attprojw = parameters.AttentionProjectionWeights.StrideToPtrAt(l);
            float* l_ln2w = parameters.LayerNorm2Weights.StrideToPtrAt(l);
            float* l_fcw = parameters.FullConnectWeights.StrideToPtrAt(l);
            float* l_fcprojw = parameters.FullConnectProjectionWeights.StrideToPtrAt(l);
            // get the pointers of the gradients of the weights for this layer
            float* dl_ln1w = grads.LayerNorm1Weights.StrideToPtrAt(l);
            float* dl_ln1b = grads.LayerNorm1Bias.StrideToPtrAt(l);
            float* dl_qkvw = grads.QKVWeights.StrideToPtrAt(l);
            float* dl_qkvb = grads.QKVBias.StrideToPtrAt(l);
            float* dl_attprojw = grads.AttentionProjectionWeights.StrideToPtrAt(l);
            float* dl_attprojb = grads.AttentionProjectionBias.StrideToPtrAt(l);
            float* dl_ln2w = grads.LayerNorm2Weights.StrideToPtrAt(l);
            float* dl_ln2b = grads.LayerNorm2Bias.StrideToPtrAt(l);
            float* dl_fcw = grads.FullConnectWeights.StrideToPtrAt(l);
            float* dl_fcb = grads.FullConnectBias.StrideToPtrAt(l);
            float* dl_fcprojw = grads.FullConnectProjectionWeights.StrideToPtrAt(l);
            float* dl_fcprojb = grads.FullConnectProjectionBias.StrideToPtrAt(l);
            // get the pointers of the activations for this layer
            float* l_ln1 = acts.LayerNorm1 + l * B * T * C;
            float* l_ln1_mean = acts.LayerNorm1Mean + l * B * T;
            float* l_ln1_rstd = acts.LayerNorm1InvStdDev + l * B * T;
            float* l_qkv = acts.QueryKeyValue + l * B * T * 3 * C;
            float* l_atty = acts.Attention + l * B * T * C;
            float* l_att = acts.PostAttention + l * B * NH * T * T;
            float* l_residual2 = acts.Residual2 + l * B * T * C;
            float* l_ln2 = acts.LayerNorm2 + l * B * T * C;
            float* l_ln2_mean = acts.LayerNorm2Mean + l * B * T;
            float* l_ln2_rstd = acts.LayerNorm2InvStdDev + l * B * T;
            float* l_fch = acts.FullyConnected + l * B * T * 4 * C;
            float* l_fch_gelu = acts.FullyConnectedGeLU + l * B * T * 4 * C;
            // get the pointers of the gradients of the activations for this layer
            float* dl_ln1 = grads_acts.LayerNorm1 + l * B * T * C;
            float* dl_qkv = grads_acts.QueryKeyValue + l * B * T * 3 * C;
            float* dl_atty = grads_acts.Attention + l * B * T * C;
            float* dl_preatt = grads_acts.PreAttention + l * B * NH * T * T;
            float* dl_att = grads_acts.PostAttention + l * B * NH * T * T;
            float* dl_attproj = grads_acts.AttentionProjected + l * B * T * C;
            float* dl_residual2 = grads_acts.Residual2 + l * B * T * C;
            float* dl_ln2 = grads_acts.LayerNorm2 + l * B * T * C;
            float* dl_fch = grads_acts.FullyConnected + l * B * T * 4 * C;
            float* dl_fch_gelu = grads_acts.FullyConnectedGeLU + l * B * T * 4 * C;
            float* dl_fcproj = grads_acts.FullyConnectedProjected + l * B * T * C;
            float* dl_residual3 = grads_acts.Residual3 + l * B * T * C;

            // backprop this layer
            llm.ResidualBackward(dl_residual3, B * T * C, dl_residual2, dl_fcproj);
            llm.MatMulBackward(dl_fcproj, l_fch_gelu, l_fcprojw, B, T, 4 * C, C, dl_fcprojw, dl_fcprojb, dl_fch_gelu);
            llm.GeLUBackward(dl_fch_gelu, l_fch, B * T * 4 * C, dl_fch);
            llm.MatMulBackward(dl_fch, l_ln2, l_fcw, B, T, C, 4 * C, dl_fcw, dl_fcb, dl_ln2);
            llm.LayerNormBackward(dl_ln2, l_residual2, l_ln2w, l_ln2_mean, l_ln2_rstd, B, T, C, dl_ln2w, dl_ln2b, dl_residual2);
            llm.ResidualBackward(dl_residual2, B * T * C, dresidual, dl_attproj);
            llm.MatMulBackward(dl_attproj, l_atty, l_attprojw, B, T, C, C, dl_attprojw, dl_attprojb, dl_atty);
            llm.AttentionBackward(dl_atty, l_att, l_qkv, B, T, C, NH, dl_preatt, dl_att, dl_qkv);
            llm.MatMulBackward(dl_qkv, l_ln1, l_qkvw, B, T, C, 3 * C, dl_qkvw, dl_qkvb, dl_ln1);
            llm.LayerNormBackward(dl_ln1, residual, l_ln1w, l_ln1_mean, l_ln1_rstd, B, T, C, dl_ln1w, dl_ln1b, dresidual);
        }
        llm.EmbedBackward(grads_acts.Embeded, inputTokenIndices, B, T, C, grads.TokenEmbeddings, grads.PositionEmbeddings);
    }

    public static unsafe void Update(Model model,
        float learningRate, float beta1, float beta2, float eps, float weightDecay, int t, TimeLlm llm)
    {
        // lazily allocate the memory for m_memory and v_memory
        if (model.m_memory == null)
        {
            model.m_memory = calloc<float>(model.Parameters.TotalCount);
            model.v_memory = calloc<float>(model.Parameters.TotalCount);
        }
        var parameters = model.Parameters.MemoryPtr;
        var gradients = model.ParameterGradients!.MemoryPtr;
        var ms = model.m_memory;
        var vs = model.v_memory;
        var parameterCount = model.Parameters.TotalCount;

        llm.Part = "3." + nameof(Update);
        llm.Index = -1;

        llm.AdamW(gradients, ms, vs, parameters, parameterCount,
                  learningRate, beta1, beta2, eps, weightDecay, t);
    }

    static unsafe TTensorPtrs AllocateAndSetPointers<TTensorPtrs>(ReadOnlySpan<nint> tensorSizes)
        where TTensorPtrs : unmanaged
    {
        ArgumentOutOfRangeException.ThrowIfNotEqual(sizeof(TTensorPtrs),
            sizeof(float*) * tensorSizes.Length);

        nint totalSize = 0;
        for (var i = 0; i < tensorSizes.Length; i++)
        {
            totalSize += tensorSizes[i];
        }
        var memory = malloc<float>(totalSize);

        var tensorPtrs = new TTensorPtrs();
        var ptrs = (float**)&tensorPtrs;
        var nextTensorPtr = memory;
        for (var i = 0; i < tensorSizes.Length; i++)
        {
            ptrs[i] = nextTensorPtr;
            nextTensorPtr += tensorSizes[i];
        }
        return tensorPtrs;
    }

    static unsafe void EnsureOutputMemory(Model model, int B, int T, int V, int L, int NH, int C)
    {
        // allocate space for all the activations if needed (done here, lazily)
        if (model.Outputs.MemoryPtr == null)
        {
            // record the current B,T as well
            model.Batchsize = B;
            model.TokenCount = T;
            // and now allocate the space
            model.OutputSizes[0] = B * T * C; // encoded
            model.OutputSizes[1] = L * B * T * C; // ln1
            model.OutputSizes[2] = L * B * T;  // ln1_mean
            model.OutputSizes[3] = L * B * T;  // ln1_rstd
            model.OutputSizes[4] = L * B * T * 3 * C; // qkv
            model.OutputSizes[5] = L * B * T * C;  // atty
            model.OutputSizes[6] = L * B * NH * T * T;  // preatt
            model.OutputSizes[7] = L * B * NH * T * T;  // att
            model.OutputSizes[8] = L * B * T * C; // attproj
            model.OutputSizes[9] = L * B * T * C; // residual2
            model.OutputSizes[10] = L * B * T * C; // ln2
            model.OutputSizes[11] = L * B * T; // ln2_mean
            model.OutputSizes[12] = L * B * T; // ln2_rstd
            model.OutputSizes[13] = L * B * T * 4 * C; // fch
            model.OutputSizes[14] = L * B * T * 4 * C; // fch_gelu
            model.OutputSizes[15] = L * B * T * C; // fcproj
            model.OutputSizes[16] = L * B * T * C; // residual3
            model.OutputSizes[17] = B * T * C; // lnf
            model.OutputSizes[18] = B * T; // lnf_mean
            model.OutputSizes[19] = B * T; // lnf_rstd
            model.OutputSizes[20] = B * T * V; // logits
            model.OutputSizes[21] = B * T * V; // probabilities
            model.OutputSizes[22] = B * T; // losses
            nint outputCount = 0;
            for (nint i = 0; i < OutputTensorCount; i++)
            {
                outputCount += model.OutputSizes[i];
            }
            model.OutputCount = outputCount;
            model.Outputs = AllocateAndSetPointers<OutputTensors>(model.OutputSizes);

            Log($"OutputCount: {outputCount}");
        }
        else
        {
            // validate B,T is no larger than what was previously allocated
            // in principle, we could re-allocate a larger chunk of memory, for now we just error output
            if (B > model.Batchsize || T > model.TokenCount)
            {
                throw new InvalidDataException("Batch size or token count is inadequately large" +
                    $"Model: B={model.Batchsize} T={model.TokenCount}, Desired: B={B} T={T}");
            }
        }
    }

    internal static unsafe void Free(Model model)
    {
        model.Parameters.Dispose();
        model.ParameterGradients?.Dispose();
        free(model.m_memory);
        free(model.v_memory);
        free(model.Outputs.MemoryPtr);
        free(model.OutputGradients.MemoryPtr);
    }

    // ----------------------------------------------------------------------------
    // data loader lite
    // returns random batches of data from a file of integers

    public unsafe class DataLoader : IDisposable
    {
        // hyperparameters
        public readonly int BatchSize;
        public readonly int TokenCount;
        // input handling and its state
        FileStream _tokensFile;
        readonly long _fileSize;
        // output memory
        public int* BatchTokenIndices;
        public int* InputTokenIndices;
        public int* TargetTokenIndices;
        // convenience variables
        public nint BatchCount;
        bool _disposedValue;

        public DataLoader(string filename, int B, int T)
        {
            BatchSize = B;
            TokenCount = T;

            // open the input file for reading
            _tokensFile = File.OpenRead(filename);
            _fileSize = _tokensFile.Length;
            if (_fileSize < (B * T + 1) * sizeof(int))
            {
                throw new InvalidDataException($"Error: file size is too small for the batch size and sequence length");
            }

            // allocate space for B*T + 1 integers to store the inputTokenIndices and targetTokenIndices
            BatchTokenIndices = malloc<int>((B * T + 1));
            InputTokenIndices = BatchTokenIndices;
            TargetTokenIndices = BatchTokenIndices + 1; // targetTokenIndices are shifted by one
            BatchCount = (nint)(_fileSize / (B * T * sizeof(int)));
        }

        public unsafe void Reset()
        {
            _tokensFile.Position = 0;
        }

        public unsafe void NextBatch()
        {
            // if we are at the end of the file, loop back to the beginning
            if (_tokensFile.Position + (BatchSize * TokenCount + 1) * sizeof(int) > _fileSize)
            {
                _tokensFile.Position = 0;
            }
            // read the B*T+1 integers from the file into batch
            _tokensFile.ReadExactlyUnmanaged(BatchTokenIndices, BatchSize * TokenCount + 1);
            //fread(this.batch, sizeof(int), B * T + 1, this.tokens_file);
            // advance the current position by B*T integers 
            //this.current_position += B * T * sizeof(int);
            // Read +1 more token to get the target and hence have to move back
            _tokensFile.Position -= sizeof(int);
        }

        public unsafe void dataloader_free()
        {
            Dispose(true);
        }

        protected virtual void Dispose(bool disposing)
        {
            if (!_disposedValue)
            {
                if (disposing)
                {
                    _tokensFile.Dispose();
                    _tokensFile = null!;
                }
                free(BatchTokenIndices);
                BatchTokenIndices = null;
                InputTokenIndices = null;
                TargetTokenIndices = null;
                _disposedValue = true;
            }
        }

        public void Dispose()
        {
            // Do not change this code. Put cleanup code in 'Dispose(bool disposing)' method
            Dispose(disposing: true);
            GC.SuppressFinalize(this);
        }
    }


    // Add the following code to the Llm class

    public unsafe static T* calloc<T>(nint size) where T : unmanaged
    {
        var ptr = malloc<T>(size);
        memset(ptr, size);
        return ptr;
    }

    public unsafe static T* malloc<T>(nint size) where T : unmanaged
    {
        return (T*)NativeMemory.Alloc((nuint)(size * sizeof(T)));
    }

    public unsafe static void free<T>(T* ptr) where T : unmanaged
    {
        NativeMemory.Free(ptr);
    }

    public unsafe static void memcpy<T>(T* dest, T* src, nint size) where T : unmanaged
    {
        var sizeInBytes = size * sizeof(T);
        Buffer.MemoryCopy(src, dest, sizeInBytes, sizeInBytes);
    }

    public unsafe static void memset<T>(T* ptr, nint size) where T : unmanaged
    {
        NativeMemory.Clear(ptr, (nuint)(size * sizeof(T)));
    }


    // ----------------------------------------------------------------------------
    // sampler

    // the GPT-2 end-of-text token id
    const int GPT2_EOT = 50256;

    static unsafe uint random_u32(ulong* state)
    {
        // xorshift rng: https://en.wikipedia.org/wiki/Xorshift#xorshift.2A
        *state ^= *state >> 12;
        *state ^= *state << 25;
        *state ^= *state >> 27;
        return (uint)((*state * 0x2545F4914F6CDD1Dul) >> 32);
    }
    static unsafe float random_f32(ulong* state)
    { // random float32 in [0,1)
        return (random_u32(state) >> 8) / 16777216.0f;
    }

    static unsafe int sample_mult(float* probabilities, int n, float coin)
    {
        // sample index from probabilities (they must sum to 1!)
        // coin is a random number in [0, 1), usually from random_f32()
        float cdf = 0.0f;
        for (int i = 0; i < n; i++)
        {
            cdf += probabilities[i];
            if (coin < cdf)
            {
                return i;
            }
        }
        return n - 1; // in case of rounding errors
    }
}
