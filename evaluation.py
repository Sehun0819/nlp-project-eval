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
    parser.add_argument(
        "--pt",
        type=str,
        help="(Our custom models only) File name of PyTorch model (.py). Should be located at model directory.",
        default="model.pt",
    )
    parser.add_argument("--temperature", type=str, default="0.2")
    parser.add_argument(
        "--task",
        type=str,
        help="Choose one of {inspect, humaneval, mercury, codexglue, fuzzing}.",
        required=True,
    )
    parser.add_argument("--lang", type=str, default="python", help="Target language")
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

    if args.task == "inspect":
        NOTE(f"For `inspect', use only one GPU `{args.gpus[0]}'")
        args.gpus = [args.gpus[0]]

    if args.task in ["humaneval", "mercury", "codexglue"]:
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


def image_name():
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


def container_home():
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


def docker_run_cmd(gpus: List[int], model: str, temperature: str, task: str, mode: str):
    gpu_idxs = "".join([str(i) for i in gpus])
    basic_flags = [
        "--shm-size=32G",
        "--rm",
        f"--name {container_name(mode, model, temperature, task, gpu_idxs)}",
    ]

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
        "-v /usr/lib/x86_64-linux-gnu:/usr/lib/x86_64-linux-gnu",
        f"-v {home_path()}/{output_dir(task)}:{container_home()}/{container_output_dir()}",
        f"-v {home_path()}/{models_dir()}/{model}:{container_home()}/{model}:ro",
    ]

    return (
        "docker run "
        + " ".join(basic_flags)
        + " "
        + " ".join(gpu_flags)
        + " "
        + " ".join(volume_flags)
        + f" {image_name()}"
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
    pt: str,
    temperature: str,
    task: str,
    batch_size,
):

    local_pt_path = f"{home_path()}/{models_dir()}/{model}/{pt}"
    if os.path.isfile(local_pt_path):
        pt_path = f"{container_home()}/{model}/{pt}"
    else:
        pt_path = '""'

    flags = [
        f"--model {container_home()}/{model}",
        f"--pt {pt_path}",
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
        f"--model {container_home()}/{model}",
        f"--temperature {temperature}",
        f"--tasks {task}",
        "--allow_code_execution",
        f"--load_generations_path {generations_path(model, temperature, task)}",
        f"--metric_output_path {eval_path(model, temperature, task)}",
    ] + task_specific_flags(task)

    eval_cmd = "python3 main.py " + " ".join(flags)

    cmd = (
        docker_run_cmd(gpus, model, temperature, task, "eval")
        + ' sh -c "'
        + eval_cmd
        + '"'
    )
    print(cmd)
    exec_and_wait(cmd)


def inspect(gpus: List[int], model: str, pt: str, batch_size: int, seq_len: int):
    local_pt_path = f"{home_path()}/{models_dir()}/{model}/{pt}"
    if os.path.isfile(local_pt_path):
        pt_flag = [f"--pt {container_home()}/{model}/{pt}"]
    else:
        pt_flag = []

    flags = [
        f"--model {container_home()}/{model}",
        f"--batch_size {batch_size}",
        f"--seq_len {seq_len}",
    ] + pt_flag

    inspect_cmd = (
        "python3 inspect_model.py "
        + " ".join(flags)
        + f" > {container_home()}/{container_output_dir()}/{model}_b{batch_size}_s{seq_len}.txt 2>&1"
    )

    cmd = (
        docker_run_cmd(gpus, model, "", "inspect", "") + ' sh -c "' + inspect_cmd + '"'
    )
    print(cmd)
    exec_and_wait(cmd)


def mk_output_dir(task):
    dir_path = home_path() / output_dir(task)
    if not os.path.isdir(dir_path):
        os.mkdir(dir_path)

    gitignore_path = dir_path / ".gitignore"
    if not os.path.isfile(gitignore_path):
        gitignore_contents = "generations_*.json\n"
        with open(gitignore_path, "w") as f:
            f.write(gitignore_contents)


def main():
    start_time = time.time()

    args = parse_arg()

    mk_output_dir(args.task)

    if args.task == "inspect":
        inspect(args.gpus, args.model, args.pt, args.batch_size, args.seq_len)
    elif args.task in ["humaneval", "mercury"] or args.task.startswith("codexglue"):
        return_code = run_LLM(
            args.gpus,
            args.model,
            args.pt,
            args.temperature,
            args.task,
            args.batch_size,
        )
        if return_code == 0:
            eval_generations(args.gpus, args.model, args.temperature, args.task)
    else:
        ### args.task == "fuzzing" ###
        NOT_IMPLEMENTED()

    end_time = time.time()
    elapsed_time_s = end_time - start_time
    elapsed_time_m = elapsed_time_s / 60
    elapsed_time_h = elapsed_time_m / 60
    elapsed_str = f"Total Running Time : {(elapsed_time_s):.2f} sec = {(elapsed_time_m):.2f} min = {(elapsed_time_h):.2f} hr"
    print(f"\n{elapsed_str}")


if __name__ == "__main__":
    main()
