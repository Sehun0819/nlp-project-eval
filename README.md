# (Optional) Clone starcoderbase-{1,3}B

```bash
git lfs install && \
git clone https://huggingface.co/bigcode/starcoderbase-1b models/starcoderbase-1b && \
git clone https://huggingface.co/bigcode/starcoderbase-3b models/starcoderbase-3b
```

# Build evaluation harness

```bash
git submodule sync && \
git submodule update --init --recursive && \
pushd bigcode-evaluation-harness && \
make DOCKERFILE=Dockerfile build && \
popd
```

# PREREQUISITE: Locate model(s) at `models` directory

```bash
mv <path_to_model> ./models/<model_name>
```

# Inspect model size (number of params, flops)

Result will be written to `_out-inspect`.

```bash
python3 evaluation.py --model=starcoderbase-1b --task=inspect --gpus=5
```

# Evaluate a task

Task should be one of {humaneval, mercury, codexglue, fuzzing}.
Result will be written to `_out-<task>`.

```bash
python3 evaluation.py --model=starcoderbase-1b --task=humaneval --gpus=5
```
