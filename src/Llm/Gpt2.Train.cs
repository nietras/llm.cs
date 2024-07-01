using System.Collections.Generic;
using System.Diagnostics;
using System.IO;

namespace nietras.LargeLanguageModel;

#pragma warning disable IDE0007 // Use implicit type

internal static partial class Gpt2
{
    internal const string ModelBinaryFileName = "gpt2_124M.bin";
    internal const string ModelDebugBinaryFileName = "gpt2_124M_debug_state.bin";

    internal const string TokenizerBinaryFileName = "gpt2_tokenizer.bin";

    internal const string DataTinyStoriesTrainBinaryFileName = "TinyStories_train.bin";
    internal const string DataTinyStoriesValidationBinaryFileName = "TinyStories_val.bin";

    internal const string TinyShakespeareTrainBinaryFileName = "tiny_shakespeare_train.bin";
    internal const string TinyShakespeareValidationBinaryFileName = "tiny_shakespeare_val.bin";

    internal static readonly IReadOnlyList<string> FileNames = [
        ModelBinaryFileName,
        ModelDebugBinaryFileName,
        TokenizerBinaryFileName,
        //DataTinyStoriesTrainBinaryFileName,
        //DataTinyStoriesValidationBinaryFileName,
        TinyShakespeareTrainBinaryFileName,
        TinyShakespeareValidationBinaryFileName];

    internal static string RemoteUrl(string fileName) =>
        @$"https://huggingface.co/datasets/nietras/llm.bin/resolve/main/{fileName}?download=true";

    // ----------------------------------------------------------------------------
    // main training loop
    public static unsafe void Train(string dataDirectory, ILlm llmToUse)
    {
        // build the GPT-2 model from a checkpoint
        Model model = BuildFromCheckpoint(dataDirectory + ModelBinaryFileName);

        // build the DataLoaders from tokens files. for now use tiny_shakespeare if available, else tiny_stories
        var tiny_stories_train = dataDirectory + DataTinyStoriesTrainBinaryFileName;
        var tiny_stories_val = dataDirectory + DataTinyStoriesValidationBinaryFileName;
        var tiny_shakespeare_train = dataDirectory + TinyShakespeareTrainBinaryFileName;
        var tiny_shakespeare_val = dataDirectory + TinyShakespeareValidationBinaryFileName;
        var train_tokens = File.Exists(tiny_shakespeare_train) ? tiny_shakespeare_train : tiny_stories_train;
        var val_tokens = File.Exists(tiny_shakespeare_val) ? tiny_shakespeare_val : tiny_stories_val;
        int B = 4; // batch size 4 (i.e. 4 independent token sequences will be trained on)
        int T = 64; // sequence length 64 (i.e. each sequence is 64 tokens nint). must be <= maxT, which is 1024 for GPT-2
        using DataLoader trainLoader = new(train_tokens, B, T);
        Log($"Train dataset BatchCount: {trainLoader.BatchCount}");

        using DataLoader validLoader = new(val_tokens, B, T);
        Log($"Valid dataset BatchCount: {validLoader.BatchCount}");
        int validBatchCount = 10;

        // some memory for generating samples from the model
        ulong rng_state = 1337;
        // during inference step we'll generate sequences of this many tokens
        const int gen_max_length = 64;
        int* gen_tokens = stackalloc int[gen_max_length];

        // train
        var stopwatch = new Stopwatch();
        var llm = CreateTimeLlm(llmToUse);
        for (int step = 0; step <= 20; step++)
        {
            // once in a while estimate the validation loss
            if (step % 10 == 0)
            {
                float validLoss = 0.0f;
                validLoader.Reset();
                for (int i = 0; i < validBatchCount; i++)
                {
                    validLoader.NextBatch();
                    var valildBatchLoss = Forward(model, validLoader.InputTokenIndices, validLoader.TargetTokenIndices, B, T, llm);
                    validLoss += valildBatchLoss;
                }
                validLoss /= validBatchCount;
                Log($"Valid loss {validLoss}");
            }

            // do a training step
            // TODO: Abstract loader and add to step perhaps and part of timings)
            trainLoader.NextBatch();
            var (loss, timings) = TrainStep(model, trainLoader.InputTokenIndices, trainLoader.TargetTokenIndices, B, T, llm, step);
            Log($"step {step}: train loss {loss} ({timings.ToReport()})");

            // once in a while do model inference to print generated text
            if (step > 0 && step % 10 == 0)
            {
                gen_tokens[0] = GPT2_EOT; // the GPT-2 EOT token kicks off the generation
                for (int t = 1; t < gen_max_length; t++)
                {
                    // note that inference is wasteful here because
                    // for each t, we re-compute all activations between 0 and t
                    // leaving this alone because you want separate code for inference anyway
                    // the inference here is just for sanity checking purposes
                    Forward(model, gen_tokens, null, 1, t, llm);
                    float* probabilities = model.Outputs.Probabilities + (t - 1) * model.Config.VocabularySize;
                    float coin = random_f32(&rng_state);
                    int next_token = sample_mult(probabilities, model.Config.VocabularySize, coin);
                    gen_tokens[t] = next_token;
                }
                Log("generated: ");
                for (int t = 0; t < gen_max_length; t++)
                {
                    LogNoNewLine($"{gen_tokens[t]} ");
                }
                Log("");
            }
        }

        Free(model);
    }
}
