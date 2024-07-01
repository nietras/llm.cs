using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;

namespace nietras.LargeLanguageModel;

#pragma warning disable IDE0007 // Use implicit type

internal static partial class Gpt2
{
    public unsafe struct ExpectedTensors
    {
        public int BatchSize;
        public int TokenCount;
        public int* InputTokenIndices;
        public int* OutputTokenIndices;
        public float* ExpectedLogits;
        public float* ExpectedLoss;
    }

    public static unsafe void Test(string dataDirectory, ILlm llmToUse, int steps, Action<string>? log)
    {
        // build the GPT-2 model from a checkpoint
        var model = BuildFromCheckpoint(dataDirectory + ModelBinaryFileName);
        int vocabularySize = model.Config.VocabularySize;
        int channelCount = model.Config.ChannelCount;
        int maxTokenCount = model.Config.MaxTokenCount;
        int layerCount = model.Config.LayerCount;

        var (inputsOutputs, expectedGrads) = ReadExpectedState(model, dataDirectory);

        log?.Invoke("[State]");
        log?.Invoke($"BatchSize: {inputsOutputs.BatchSize}");
        log?.Invoke($"TokenCount: {inputsOutputs.TokenCount}");

        // expected losses are as follows, from Python
        float[] expectedLosses = [
            5.270007133483887f,
            4.059706687927246f,
            3.3751230239868164f,
            2.8007826805114746f,
            2.315382242202759f,
            1.8490285873413086f,
            1.3946564197540283f,
            0.9991465210914612f,
            0.6240804195404053f,
            0.37651097774505615f
        ];

        // overall OK signal for the test
        bool allOk = true;

        // training iterations, following the pytorch code
        float* losses = stackalloc float[steps];
        var llm = CreateTimeLlm(llmToUse);
        for (int step = 0; step < steps; step++)
        {
            var timingEnabled = step >= JitAndWarmupCount;
            llm.Enabled = timingEnabled;

            var (loss, t) = TrainStep(model,
                inputsOutputs.InputTokenIndices, inputsOutputs.OutputTokenIndices,
                inputsOutputs.BatchSize, inputsOutputs.TokenCount,
                llm, step);

            // error checking at step 0 for reference activations/gradients
            if (step == 0)
            {
                // at this point
                var logitsOk = CheckTensor(inputsOutputs.ExpectedLogits, model.Outputs.Logits,
                    inputsOutputs.BatchSize * inputsOutputs.TokenCount * vocabularySize, "Logits");

                var gradsOk = CheckTensors(model.ParameterGradients!, expectedGrads);
                allOk &= logitsOk && gradsOk;
            }
            losses[step] = loss;
            var warmupMessage = timingEnabled ? "" : " JIT/WARMUP";
            if (step < expectedLosses.Length)
            {
                var expectedLoss = expectedLosses[step];
                var lossOk = CheckLoss(loss, expectedLoss);
                allOk = allOk && lossOk;
                log?.Invoke($"{step,2}: loss {loss:F6} exp. {expectedLoss:F6} {(lossOk ? "OK" : "FAIL"),-4} " +
                            $"({t.ToReport()}){warmupMessage}");
            }
            else
            {
                log?.Invoke($"{step,2}: loss {loss:F6} ({t.ToReport()}){warmupMessage}");
            }
        }
        log?.Invoke($"All okay: {allOk}");

        var timeReport = llm.CreateReport(steps - JitAndWarmupCount);

        log?.Invoke(timeReport);

        // free everything
        free(inputsOutputs.InputTokenIndices);
        free(inputsOutputs.OutputTokenIndices);
        free(inputsOutputs.ExpectedLogits);
        free(inputsOutputs.ExpectedLoss);
        free(expectedGrads.MemoryPtr);
        Free(model);

        if (!allOk) { throw new ArithmeticException($"{llmToUse.GetType().Name} failed {nameof(Gpt2)} train test run, see output for details."); }
    }

    internal static unsafe (ExpectedTensors ExpectedInputsOutputs, ParameterTensors ExpectedGrads)
        ReadExpectedState(in Model model, string dataDirectory)
    {
        using var stateFile = File.OpenRead(dataDirectory + ModelDebugBinaryFileName);

        var expectedInputsOutputs = ReadExpectedTensors(model.Config.VocabularySize, stateFile);

        var expectedGrads = ParameterTensors.Create(model.Config);
        stateFile.ReadExactlyUnmanaged(expectedGrads.MemoryPtr, expectedGrads.TotalCount);

        return (expectedInputsOutputs, expectedGrads);
    }

    internal static unsafe ExpectedTensors ReadExpectedTensors(int vocabularySize, FileStream stateFile)
    {
        Span<int> state_header = stackalloc int[256];
        // read span from model_file
        stateFile.ReadExactlyUnmanaged(state_header);

        //fread(model_header, sizeof(int), 256, model_file);
        if (state_header[0] != 20240327) { throw new InvalidDataException($"Bad magic model file"); }
        if (state_header[1] != 1) { throw new InvalidDataException($"Bad version in model file"); }
        int batchSize = state_header[2]; // batch size, e.g. 4
        int tokenCount = state_header[3]; // time/tokenCount / sequence length (e.g. 64, up to maxT)

        // inputs and expected outputs, only used for error checking
        int* x = malloc<int>(batchSize * tokenCount);
        int* y = malloc<int>(batchSize * tokenCount);
        float* expectedLogits = malloc<float>(batchSize * tokenCount * vocabularySize);
        float* expectedLoss = malloc<float>(1);

        // read reference information from Python
        stateFile.ReadExactlyUnmanaged(x, batchSize * tokenCount);
        stateFile.ReadExactlyUnmanaged(y, batchSize * tokenCount);
        stateFile.ReadExactlyUnmanaged(expectedLogits, batchSize * tokenCount * vocabularySize);
        stateFile.ReadExactlyUnmanaged(expectedLoss, 1);

        var expectedInputsOutputs = new ExpectedTensors()
        {
            BatchSize = batchSize,
            TokenCount = tokenCount,
            InputTokenIndices = x,
            OutputTokenIndices = y,
            ExpectedLogits = expectedLogits,
            ExpectedLoss = expectedLoss,
        };
        return expectedInputsOutputs;
    }

    static unsafe bool CheckTensors(
        IReadOnlyList<Tensor<float>> grads,
        IReadOnlyList<Tensor<float>> expectedGrads)
    {
        var allOk = grads.Count == expectedGrads.Count;
        for (int i = 0; i < grads.Count; i++)
        {
            var grad = grads[i];
            var expectedGrad = expectedGrads[i];
            Debug.Assert(grad.Name == expectedGrad.Name);
            Debug.Assert(grad.Count == expectedGrad.Count);
            var ok = CheckTensor(grad.Ptr, expectedGrad.Ptr, grad.Count, grad.Name);
            allOk = allOk && ok;
        }
        return allOk;
    }

    const float CheckDiffLimit = 0.01f;
    static bool CheckLoss(float a, float b) => Check(a, b);
    static bool Check(float a, float b) => MathF.Abs(a - b) < CheckDiffLimit;

    // poor man's tensor checker
    static unsafe bool CheckTensor(float* actual, float* expected, nint count, string label)
    {
        const int printUpTo = 0;//5;
        LogNoNewLine($"{label,-28} ");
        bool ok = true;
        var maxAbsDiff = 0f;
        for (nint i = 0; i < count; i++)
        {
            var a = actual[i];
            var e = expected[i];

            var absDiff = MathF.Abs(a - e);
            maxAbsDiff = MathF.Max(absDiff, maxAbsDiff);

            var isOk = absDiff < CheckDiffLimit;
            ok &= isOk;
            if (i < printUpTo)
            {
                Log("");
                LogNoNewLine($"{(isOk ? "OK  " : "FAIL")} {a,15} {e,15} {absDiff,15}");
            }
            if (!isOk) { Debugger.Break(); }
        }
        Log($"TENSOR {(ok ? "OK  " : "FAIL")} MaxAbsDiff {maxAbsDiff,8:F6}");
        return ok;
    }
}
