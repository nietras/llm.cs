using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace nietras.LargeLanguageModel.Test;

[TestClass]
public class DevTest
{
    [TestMethod]
    public void DevTest_()
    {
        Gpt2.Config c = new() { VocabularySize = 16, MaxTokenCount = 32, LayerCount = 4, ChannelCount = 8, HeadCount = 16 };
        var t = Gpt2.ParameterTensorsNew.Create(c);

    }
}
