import pytest
import torch

from moss_tts.resampling import StreamingResampler, output_resampler


@pytest.mark.parametrize("length,chunks", [(1, 1), (33, 7), (1920 * 3, 1920), (8000, 113)])
def test_chunk_boundaries_match_whole_signal(length, chunks):
    torch.manual_seed(42)
    waveform = torch.randn(length)
    kernel = output_resampler()
    stream = StreamingResampler(kernel)
    pieces = [stream.push(chunk) for chunk in waveform.split(chunks)]
    pieces.append(stream.flush())
    actual = torch.cat(pieces)
    assert actual.numel() == 2 * length
    torch.testing.assert_close(actual, kernel(waveform), atol=1e-6, rtol=1e-5)
    assert stream.flush().numel() == 0
    with pytest.raises(RuntimeError, match="flushed"):
        stream.push(waveform)


def test_pitch_duration_and_stopband():
    rate = 24000
    time = torch.arange(rate, dtype=torch.float32) / rate
    waveform = torch.sin(2 * torch.pi * 1000 * time)
    stream = StreamingResampler(output_resampler())
    output = torch.cat([*(stream.push(chunk) for chunk in waveform.split(1920)), stream.flush()])
    assert output.numel() / 48000 == waveform.numel() / rate
    spectrum = torch.fft.rfft(output)
    assert int(spectrum.abs().argmax()) == 1000
    assert spectrum[23000].abs() / spectrum[1000].abs() < 1e-4


def test_empty_stream_has_no_audio():
    stream = StreamingResampler(output_resampler())
    assert stream.flush().numel() == 0
