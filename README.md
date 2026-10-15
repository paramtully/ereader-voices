# eReader voices

The voice files the eReader app downloads on first use: Kokoro-82M as an ONNX model, its voice styles, and the pronunciation data its phonemizer needs. They are published as release assets so the app fetches them from a place under its own control.

## Release kokoro-v1.0-1

Each asset is a copy of the file at the given path and revision of its origin repository, except `model_fp16.onnx`, which `patch_atan.py` changes (see below). `manifest.json`, also an asset, lists each file's size and SHA-256.

| Asset | Bytes | Origin | Revision | Path there | License |
|---|---:|---|---|---|---|
| `model_fp16.onnx` (patched) | 163,233,198 | [onnx-community/Kokoro-82M-v1.0-ONNX-timestamped](https://huggingface.co/onnx-community/Kokoro-82M-v1.0-ONNX-timestamped) | `dd4401a9add81ac692d20e240d22ec9dda82cc29` | `onnx/model_fp16.onnx` | Apache 2.0 [1] [2] |
| `tokenizer.json` | 3,497 | onnx-community/Kokoro-82M-v1.0-ONNX-timestamped | `dd4401a9add81ac692d20e240d22ec9dda82cc29` | `tokenizer.json` | Apache 2.0 [1] [2] |
| `af_heart.bin` | 522,240 | onnx-community/Kokoro-82M-v1.0-ONNX-timestamped | `dd4401a9add81ac692d20e240d22ec9dda82cc29` | `voices/af_heart.bin` | Apache 2.0 [1] [2] |
| `af_bella.bin` | 522,240 | onnx-community/Kokoro-82M-v1.0-ONNX-timestamped | `dd4401a9add81ac692d20e240d22ec9dda82cc29` | `voices/af_bella.bin` | Apache 2.0 [1] [2] |
| `af_nicole.bin` | 522,240 | onnx-community/Kokoro-82M-v1.0-ONNX-timestamped | `dd4401a9add81ac692d20e240d22ec9dda82cc29` | `voices/af_nicole.bin` | Apache 2.0 [1] [2] |
| `am_michael.bin` | 522,240 | onnx-community/Kokoro-82M-v1.0-ONNX-timestamped | `dd4401a9add81ac692d20e240d22ec9dda82cc29` | `voices/am_michael.bin` | Apache 2.0 [1] [2] |
| `am_fenrir.bin` | 522,240 | onnx-community/Kokoro-82M-v1.0-ONNX-timestamped | `dd4401a9add81ac692d20e240d22ec9dda82cc29` | `voices/am_fenrir.bin` | Apache 2.0 [1] [2] |
| `bf_emma.bin` | 522,240 | onnx-community/Kokoro-82M-v1.0-ONNX-timestamped | `dd4401a9add81ac692d20e240d22ec9dda82cc29` | `voices/bf_emma.bin` | Apache 2.0 [1] [2] |
| `bm_george.bin` | 522,240 | onnx-community/Kokoro-82M-v1.0-ONNX-timestamped | `dd4401a9add81ac692d20e240d22ec9dda82cc29` | `voices/bm_george.bin` | Apache 2.0 [1] [2] |
| `misaki-en_us-g2p.onnx` | 3,438,244 | [nekwo/misaki-en-us-g2p-onnx](https://huggingface.co/nekwo/misaki-en-us-g2p-onnx) | `4b3a605d2f605318ab4f762ab9855c44106528b7` | `misaki-en_us-g2p.onnx` | Apache 2.0 [3] [4] |
| `misaki-en_us-g2p.json` | 5,724,857 | nekwo/misaki-en-us-g2p-onnx | `4b3a605d2f605318ab4f762ab9855c44106528b7` | `misaki-en_us-g2p.json` | Apache 2.0 [3] [5] |

`misaki-en_us-g2p.onnx` is the grapheme-to-phoneme model misaki falls back on for words its dictionaries lack. `misaki-en_us-g2p.json` holds misaki's `us_gold` and `us_silver` dictionaries, the model's letter and phoneme tables, and Kokoro's phoneme vocabulary.

Licenses, as each origin states them:

1. Kokoro-82M weights and voices: Apache 2.0, the `license: apache-2.0` field of the [hexgrad/Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M) model card.
2. The ONNX export: Apache 2.0, the `license: apache-2.0` field of the [onnx-community model card](https://huggingface.co/onnx-community/Kokoro-82M-v1.0-ONNX-timestamped/blob/dd4401a9add81ac692d20e240d22ec9dda82cc29/README.md) at the revision above.
3. The G2P package: Apache 2.0, the `license: apache-2.0` field of the [nekwo model card](https://huggingface.co/nekwo/misaki-en-us-g2p-onnx/blob/4b3a605d2f605318ab4f762ab9855c44106528b7/README.md) at the revision above; its "Credits and licence" section releases the ONNX conversion and packaging under Apache-2.0.
4. The G2P model's weights: Apache 2.0, the `license: apache-2.0` field of the [PeterReid/graphemes_to_phonemes_en_us](https://huggingface.co/PeterReid/graphemes_to_phonemes_en_us) model card, which the nekwo card names as its source.
5. misaki's dictionaries: Apache 2.0, the [LICENSE](https://github.com/hexgrad/misaki/blob/fba1236595f2d2bf21d414ba6e57d25256afada3/LICENSE) of hexgrad/misaki at `fba1236595f2`, the commit the nekwo card says the data was built from.

The full license text is in [LICENSES/Apache-2.0.txt](LICENSES/Apache-2.0.txt), and [LICENSES/NOTICE](LICENSES/NOTICE) names each redistributed work, its origin and its copyright holder. Every redistributed work is under Apache 2.0, so no other license text is needed.

## The patch to model_fp16.onnx

The half precision export returns a waveform of NaN, heard as silence, for about one piece of text in eight. The generator takes the phase of its harmonic source's STFT with atan2, which the export writes as `Atan(Div(imag, real))`; a bin of that STFT can be exactly 0+0i in half precision, the Div computes 0/0, and the NaN spreads to the whole waveform.

`patch_atan.py` inserts `Where(IsNaN(q), 0, q)` between that Div and the Atan. The quotient is NaN only for 0/0, which now gives a phase of 0, as atan2(0, 0) does in PyTorch; every other quotient passes through unchanged, so the model's output is the same wherever it was not NaN. The opset (20), IR version (9) and all other nodes are as exported.

To rebuild the asset from the original (sha256 `220724d5c5e0cc01be30f38faa6cf0c895a7cde6e7773e91db2973c8c7e5123c`):

```
pip install onnx==1.23.2 numpy
python patch_atan.py model_fp16.onnx patched/model_fp16.onnx
```

With onnx 1.23.2 the output matches the asset byte for byte (sha256 `5e65f5f993446148e0b1e35925ba110afb91cd4e90f1c3e8aaf303dca38b1d76`).
