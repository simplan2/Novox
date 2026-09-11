using ElBruno.QwenTTS.Pipeline;
using Microsoft.Extensions.DependencyInjection;

namespace ElBruno.QwenTTS.Core.Tests;

/// <summary>
/// Tests that QwenTextToSpeechClient and DI registration correctly handle
/// variant parameters. Validates constructor acceptance, DI with 1.7B variant,
/// and the CreateAsync variant parameter flows through TtsPipeline.
/// </summary>
public class VariantClientIntegrationTests
{
    // ── Constructor Variant Acceptance ───────────────────────────────

    [Fact]
    public void Client_DefaultConstructor_Uses06B()
    {
        using var client = new QwenTextToSpeechClient();
        // Should not throw — defaults to 0.6B
        Assert.NotNull(client);
    }

    [Fact]
    public void Client_With17BVariant_Constructs()
    {
        using var client = new QwenTextToSpeechClient(
            variant: QwenModelVariant.Qwen17B);
        Assert.NotNull(client);
    }

    [Fact]
    public void Client_With17BAndInstruct_Constructs()
    {
        using var client = new QwenTextToSpeechClient(
            variant: QwenModelVariant.Qwen17B,
            modelDir: @"C:\fake\models");
        Assert.NotNull(client);
    }

    [Fact]
    public async Task Client_With17B_DisposeThenCall_ThrowsDisposed()
    {
        var client = new QwenTextToSpeechClient(variant: QwenModelVariant.Qwen17B);
        client.Dispose();

        await Assert.ThrowsAsync<ObjectDisposedException>(
            () => client.SynthesizeToMemoryAsync("Hello"));
    }

    // ── DI Registration with Variant ────────────────────────────────

    [Fact]
    public void DI_AddQwenTextToSpeechClient_With17B_Registers()
    {
        var services = new ServiceCollection();
        services.AddQwenTextToSpeechClient(opts =>
        {
            opts.ModelVariant = QwenModelVariant.Qwen17B;
            opts.InstructText = "Read with warmth";
        });

        var descriptor = Assert.Single(services, d => d.ServiceType == typeof(ITextToSpeechClient));
        Assert.Equal(ServiceLifetime.Singleton, descriptor.Lifetime);
    }

    [Fact]
    public void DI_AddQwenTts_With17B_Registers()
    {
        var services = new ServiceCollection();
        services.AddQwenTts(opts =>
        {
            opts.ModelVariant = QwenModelVariant.Qwen17B;
        });

        var descriptor = Assert.Single(services, d => d.ServiceType == typeof(ITtsPipeline));
        Assert.Equal(ServiceLifetime.Singleton, descriptor.Lifetime);
    }

    // ── CreateAsync with variant ────────────────────────────────────

    [Fact]
    public async Task CreateAsync_With17B_NullModelDir_ResolvesVariantDefaults()
    {
        // Cancel immediately to prevent download, but verify it resolves to 1.7B paths
        using var cts = new CancellationTokenSource();
        cts.Cancel();

        await Assert.ThrowsAsync<OperationCanceledException>(() =>
            TtsPipeline.CreateAsync(
                modelDir: null,
                variant: QwenModelVariant.Qwen17B,
                cancellationToken: cts.Token));
    }

    [Fact]
    public async Task CreateAsync_With06B_NullModelDir_ResolvesLegacyDefaults()
    {
        using var cts = new CancellationTokenSource();
        cts.Cancel();

        await Assert.ThrowsAsync<OperationCanceledException>(() =>
            TtsPipeline.CreateAsync(
                modelDir: null,
                variant: QwenModelVariant.Qwen06B,
                cancellationToken: cts.Token));
    }

    [Fact]
    public async Task CreateAsync_DetailedProgress_With17B_AcceptsVariant()
    {
        using var cts = new CancellationTokenSource();
        cts.Cancel();

        var progress = new Progress<ModelDownloadProgress>(_ => { });

        await Assert.ThrowsAsync<OperationCanceledException>(() =>
            TtsPipeline.CreateAsync(
                modelDir: null,
                downloadProgress: progress,
                variant: QwenModelVariant.Qwen17B,
                cancellationToken: cts.Token));
    }
}
