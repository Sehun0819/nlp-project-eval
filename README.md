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

# (not implemented) Inspect model size

Result will be written to `_out-inspect/{.txt}`.

```bash
python3 evaluation.py --model=starcoderbase-1b --inspect --gpus=2,3
```

# Evaluate a task

Result will be written to `_out-<task>/{.json, .txt}`.

```bash
python3 evaluation.py --model=starcoderbase-1b --task=humaneval --gpus=2,3
```
