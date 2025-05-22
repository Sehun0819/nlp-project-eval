import argparse
import os
import subprocess
import pathlib
import time
from typing import List


def emphasize(s):
    line = "+" + ("-" * (len(s) + 2)) + "+\n"
    return line + "| " + s + " |\n" + line


def NOTE(msg):
    print(emphasize(msg))


def CHECK(b, msg):
    if not b:
        print(emphasize(f"ERROR: {msg}"))
        exit(0)


def NOT_IMPLEMENTED():
    print("ERROR: Not Implemented")
    exit(1)


def parse_arg():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        type=str,
        help="Directory name of target model. Should be located at this directory.",
        required=True,
    )
    parser.add_argument("--temperature", type=str, default="1")
    parser.add_argument(
        "--task",
        type=str,
        help="Choose one of {inspect, humaneval, mercury, codexglue, fuzzing}.",
        required=True,
    )
    parser.add_argument("--lang", type=str, help="Target language")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--seq_len", type=int, default=128)
    parser.add_argument(
        "--gpus", type=str, help="Comma seperated indexes of GPU(s) to use"
    )

    args = parser.parse_args()

    if args.gpus is not None:
        args.gpus = [int(s.strip()) for s in args.gpus.split(",")]

    allowed_tasks = ["inspect", "humaneval", "mercury", "codexglue", "fuzzing"]
    CHECK(
        args.task in allowed_tasks,
        f"Task must be one of {allowed_tasks}, received `{args.task}'",
    )

    if args.task == "inspect" and len(args.gpus) > 1:
        NOTE(f"For `inspect', use only one GPU `{args.gpus[0]}'")
        args.gpus = [args.gpus[0]]

    if args.task in ["humaneval", "mercury", "codexglue"]:
        if not args.lang:
            args.lang = "python"
        if args.task == "humaneval" or args.task == "mercury":
            allowed_langs = ["python"]
        elif args.task == "codexglue":
            allowed_langs = ["python", "go", "java", "javascript", "php", "ruby"]
        else:  ### args.task == "fuzzing" ###
            allowed_langs = ["c", "python", "java", "go"]
        CHECK(
            args.lang in allowed_langs,
            f"For {args.task}, target language must be one of {allowed_langs}, received `{args.lang}'.",
        )

        if args.task == "mercury":
            CHECK(
                len(args.gpus) == 1,
                f"For `mercury', provide only one GPU to prevent communication error. (provided GPUs={args.gpus})",
            )

        if args.task == "codexglue":
            args.task = f"codexglue_code_to_text-{args.lang}"

    if args.task == "fuzzing":
        if not args.lang:
            args.lang = "cpp"
        allowed_langs = ["c", "cpp", "java", "go", "python"]
        CHECK(
            args.lang in allowed_langs,
            f"For {args.task}, target language must be one of {allowed_langs}, received `{args.lang}'.",
        )

    if args.model == "starcoderbase-3b" and (
        args.task == "mercury" or args.task.startswith("codexglue")
    ):
        bs = 32
        if args.batch_size > bs:
            NOTE(f"Set `batch_size' to {bs}, for preventing CUDA out of memory.")
            args.batch_size = bs

    return args


def home_path():
    return pathlib.Path(__file__).parent.resolve()


def models_dir():
    return "models"


def output_dir(task):
    return f"_out-{task}"


def fuzzer_output_dir(model, lang, temperature):
    return f"{model}_{lang}_t{temperature}"


def fuzzer_corpus_dir():
    return f"corpus"


def image_name(task):
    if task == "fuzzing":
        return "fuzz4all:nlp-project-eval"
    else:
        return "evaluation-harness"


def container_name(mode, model, temperature, task, gpus):
    if mode:
        name = f"{mode}-{model}"
    else:
        name = f"{model}"

    if temperature:
        name += f"-t{temperature}-{task}-gpus{gpus}"
    else:
        name += f"-{task}-gpus{gpus}"

    return name


def container_home(task):
    if task == "fuzzing":
        return "/root"
    else:
        return "/app"


def container_output_dir():
    return "output"


def generations_path(model: str, temperature: str, task: str = ""):
    if task == "":
        return f"{container_output_dir()}/generations_{model}_{temperature}.json"
    else:
        return f"{container_output_dir()}/generations_{model}_{temperature}_{task}.json"


def eval_path(model: str, temperature: str, task: str):
    return f"{container_output_dir()}/eval_{model}_{temperature}_{task}.json"


def exec_and_wait(cmd, work_dir=home_path(), env=os.environ.copy()):
    cwd = os.getcwd()
    os.chdir(work_dir)
    proc = subprocess.Popen([cmd], shell=True)
    _ = proc.communicate()
    os.chdir(cwd)
    return proc.returncode


def docker_run_cmd(
    gpus: List[int],
    model: str,
    temperature: str,
    task: str,
    mode: str,
    workdir: str = None,
    additional_flags: List[str] = [],
):
    gpu_idxs = "".join([str(i) for i in gpus])
    workdir_flag = [f"-w {workdir}"] if workdir else []
    basic_flags = (
        [
            "--shm-size=32G",
            "--rm",
            f"--name {container_name(mode, model, temperature, task, gpu_idxs)}",
        ]
        + workdir_flag
        + additional_flags
    )

    device_str = "'\"device=" + ",".join([str(i) for i in gpus]) + "\"'"
    gpu_flags = [
        "--runtime=nvidia",
        f"--gpus {device_str}",
        "--device=/dev/nvidia-uvm",
        "--device=/dev/nvidia-uvm-tools",
        "--device=/dev/nvidia-modeset",
        "--device=/dev/nvidiactl",
    ] + [f"--device=/dev/nvidia{i}" for i in gpus]

    volume_flags = ["-v /usr/local/cuda-12.4:/usr/local/cuda"] if len(gpus) > 0 else []
    volume_flags += [
        # "-v /usr/lib/x86_64-linux-gnu:/usr/lib/x86_64-linux-gnu",
        f"-v {home_path()}/{output_dir(task)}:{container_home(task)}/{container_output_dir()}",
        f"-v {home_path()}/{models_dir()}/{model}:{container_home(task)}/{model}:ro",
    ]

    return (
        "docker run "
        + " ".join(basic_flags)
        + " "
        + " ".join(gpu_flags)
        + " "
        + " ".join(volume_flags)
        + f" {image_name(task)}"
    )


def task_specific_flags(task):
    if task == "humaneval":
        return [
            "--max_length_generation 650",
            "--n_samples 200",
        ]
    elif task == "mercury":
        return [
            # "--load_in_4bit",
            "--max_length_generation 2048",
            "--n_samples 5",
        ]
    elif task.startswith("codexglue"):
        return [
            "--max_length_generation 2048",
            "--n_samples 1",
        ]


def run_LLM(
    gpus: List[int],
    model: str,
    temperature: str,
    task: str,
    batch_size,
):

    flags = [
        f"--model {container_home(task)}/{model}",
        f"--temperature {temperature}",
        f"--tasks {task}",
        f"--batch_size {batch_size}",
        "--trust_remote_code",
        "--generation_only",
        "--save_generations",
        f"--save_generations_path  {generations_path(model, temperature)}",
    ] + task_specific_flags(task)

    gen_cmd = "accelerate launch main.py " + " ".join(flags)

    cmd = docker_run_cmd(gpus, model, temperature, task, "run") + " " + gen_cmd
    print(cmd)
    return exec_and_wait(cmd)


def eval_generations(gpus: List[int], model: str, temperature: str, task: str):
    flags = [
        f"--model {container_home(task)}/{model}",
        f"--temperature {temperature}",
        f"--tasks {task}",
        "--allow_code_execution",
        f"--load_generations_path {generations_path(model, temperature, task)}",
        f"--metric_output_path {eval_path(model, temperature, task)}",
    ] + task_specific_flags(task)

    eval_cmd = "python3 -u main.py " + " ".join(flags)

    cmd = (
        docker_run_cmd(gpus, model, temperature, task, "eval")
        + ' sh -c "'
        + eval_cmd
        + '"'
    )
    print(cmd)
    exec_and_wait(cmd)


def inspect(gpus: List[int], model: str, batch_size: int, seq_len: int):
    task = "inspect"

    flags = [
        f"--model {container_home(task)}/{model}",
        f"--batch_size {batch_size}",
        f"--seq_len {seq_len}",
    ]

    inspect_cmd = (
        "python3 -u inspect_model.py "
        + " ".join(flags)
        + f" > {container_home(task)}/{container_output_dir()}/{model}_b{batch_size}_s{seq_len}.txt 2>&1"
    )

    cmd = docker_run_cmd(gpus, model, "", task, "") + ' sh -c "' + inspect_cmd + '"'
    print(cmd)
    exec_and_wait(cmd)


def fuzzing(
    gpus: List[int],
    model: str,
    temperature: str,
    lang: str,
    batch_size: int,
):
    task = "fuzzing"

    configs_dir = "fuzzing_configs"
    container_configs_dir = "/configs"

    if lang == "c":
        target_name = "/home/gcc-13/bin/gcc"
        config = f"{container_configs_dir}/c_std.yaml"
        cov_dir = "C"
    elif lang == "cpp":
        target_name = "/home/gcc-13/bin/g++"
        config = f"{container_configs_dir}/cpp_23.yaml"
        cov_dir = "CPP"
    # elif lang == "go":
    #     target_name = "/home/go/bin/go"
    #     config = "config/half_run/go_std.yaml"
    #     cov_dir = "GO"
    # elif lang == "java":
    #     target_name = "/home/javac/bin/javac"
    #     config = "config/half_run/java_std.yaml"
    #     cov_dir = "JAVA"
    # elif lang == "python":
    #     target_name = "python"  # just python is enough.
    #     config = "config/half_run/qiskit_opt_and_qasm.yaml"
    #     cov_dir = "QISKIT"
    else:
        NOT_IMPLEMENTED()

    container_output_path = f"{container_home(task)}/{container_output_dir()}/{fuzzer_output_dir(model, lang, temperature)}"
    container_corpus_path = f"{container_output_path}/{fuzzer_corpus_dir()}"

    flags = [
        f"--config {config} main_with_config",
        f"--folder {container_corpus_path}",
        f"--batch_size {batch_size}",
        f"--model_name {container_home(task)}/{model}",
        f"--target {target_name}",
    ]

    fuzz_cmd = (
        "/root/anaconda3/bin/conda run -n fuzz4all python3 -u Fuzz4All/fuzz.py "
        + " ".join(flags)
        + f" > {container_output_path}/fuzz_{model}_{lang}_t{temperature}.txt 2>&1"
    )

    cmd = (
        docker_run_cmd(
            gpus,
            model,
            temperature,
            "fuzzing",
            "fuzz",
            workdir="/home/Fuzz4All",
            additional_flags=[
                f"-v {home_path()}/{configs_dir}:{container_configs_dir}"
            ],
        )
        + ' sh -c "'
        + fuzz_cmd
        + '"'
    )
    print(cmd)
    exec_and_wait(cmd)

    cov_cmd = (
        f"/root/anaconda3/bin/conda run -n fuzz4all python3 -u tools/coverage/{cov_dir}/collect_coverage.py --folder {container_corpus_path}"
        + f" > {container_output_path}/cov_{model}_{lang}_t{temperature}.txt 2>&1"
    )

    cmd = (
        docker_run_cmd(
            gpus,
            model,
            temperature,
            "fuzzing",
            "cov",
            workdir="/home/Fuzz4All",
            additional_flags=[
                f"-v {home_path()}/{configs_dir}:{container_configs_dir}"
            ],
        )
        + ' sh -c "'
        + cov_cmd
        + '"'
    )
    print(cmd)
    exec_and_wait(cmd)


def mk_output_dir(task, model, lang, temperature):
    dir_path = home_path() / output_dir(task)
    if not os.path.isdir(dir_path):
        os.mkdir(dir_path)

    if task in ["humaneval", "mercury"] or task.startswith("codexglue"):
        gitignore_path = dir_path / ".gitignore"
        if not os.path.isfile(gitignore_path):
            gitignore_contents = "generations_*.json\n"
            with open(gitignore_path, "w") as f:
                f.write(gitignore_contents)
    elif task == "fuzzing":
        fuzzer_output_path = dir_path / fuzzer_output_dir(model, lang, temperature)
        if not os.path.isdir(fuzzer_output_path):
            os.mkdir(fuzzer_output_path)

        fuzzer_corpus_path = (
            dir_path / fuzzer_output_dir(model, lang, temperature) / fuzzer_corpus_dir()
        )
        if not os.path.isdir(fuzzer_corpus_path):
            os.mkdir(fuzzer_corpus_path)

        gitignore_path = fuzzer_corpus_path / ".gitignore"
        if not os.path.isfile(gitignore_path):
            gitignore_contents = "*.fuzz\n*.txt\n"
            with open(gitignore_path, "w") as f:
                f.write(gitignore_contents)


def main():
    start_time = time.time()

    args = parse_arg()

    mk_output_dir(args.task, args.model, args.lang, args.temperature)

    if args.task == "inspect":
        inspect(args.gpus, args.model, args.batch_size, args.seq_len)
    elif args.task in ["humaneval", "mercury"] or args.task.startswith("codexglue"):
        # return_code = run_LLM(
        #     args.gpus,
        #     args.model,
        #     args.temperature,
        #     args.task,
        #     args.batch_size,
        # )
        # if return_code == 0:
        #     eval_generations(args.gpus, args.model, args.temperature, args.task)
        eval_generations(args.gpus, args.model, args.temperature, args.task)
    elif args.task == "fuzzing":
        fuzzing(args.gpus, args.model, args.temperature, args.lang, args.batch_size)
    else:
        CHECK(False, f"Unsupported task `{args.task}'")

    end_time = time.time()
    elapsed_time_s = end_time - start_time
    elapsed_time_m = elapsed_time_s / 60
    elapsed_time_h = elapsed_time_m / 60
    elapsed_str = f"Total Running Time : {(elapsed_time_s):.2f} sec = {(elapsed_time_m):.2f} min = {(elapsed_time_h):.2f} hr"
    print(f"\n{elapsed_str}")


if __name__ == "__main__":
    main()
