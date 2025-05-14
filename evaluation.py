import argparse
import os
import subprocess
import pathlib
import time
from typing import List


def CHECK(b, msg):
    if not b:
        print(f"ERROR: {msg}")
        exit(0)


def NOT_IMPLEMENTED():
    print("ERROR: Not Implemented")
    exit(1)


def parse_arg():
    parser = argparse.ArgumentParser()

    # Basic Options
    parser.add_argument(
        "--model",
        type=str,
        help="Directory name of target model. Should be located at this directory.",
    )
    parser.add_argument(
        "--pt",
        type=str,
        help="(Our custom models only) File name of PyTorch model (.py). Should be located at model directory.",
        default="model.pt",
    )
    parser.add_argument(
        "--inspect", action="store_true", help="Inspect model size."
    )
    parser.add_argument(
        "--task", type=str, help="Choose one of {humaneval, mercury, codexglue, fuzzing}."
    )
    parser.add_argument(
        "--lang", type=str, default="python", help="Target language"
    )

    # GPU Option
    parser.add_argument("--gpus", type=str, help="Comma seperated indexes of GPU(s) to use")

    # (Optional) Flags
    parser.add_argument("--max_length_generation", type=str, default="650")
    parser.add_argument("--temperature", type=str, default="0.8")
    parser.add_argument("--do_sample", type=str, default="True")
    parser.add_argument("--n_samples", type=str, default="200")
    parser.add_argument("--batch_size", type=str, default="64")

    args = parser.parse_args()

    if args.gpus is not None:
        args.gpus = [s.strip() for s in args.gpus.split(",")]

    CHECK((args.inpect and not args.task) or (not args.inpect and args.task),
          "You should provide either '--inspect' or '--task=<task>'.")

    if args.task:
        allowed_tasks = ["humaneval", "mercury", "codexglue", "fuzzing"]
        CHECK(args.task in allowed_tasks, f"Task must be one of {allowed_tasks}, received `{args.task}'")

        if args.task == "humaneval" or args.task == "mercury":
            allowed_langs = ["python"]
        elif args.task == "codexglue":
            allowed_langs = ["python", "go", "java", "javascript", "php", "ruby"]
        else: ### args.task == "fuzzing" ###
            allowed_langs = ["c", "python", "java", "go"]
        CHECK(args.lang in allowed_langs, f"For {args.task}, target language must be one of {allowed_langs}, received `{args.lang}'")

        if args.task == "codexglue":
            args.task = f"codexglue_code_to_text-{args.lang}"

    return args


def home_path():
    return pathlib.Path(__file__).parent.resolve()


def models_dir():
    return "models"


def output_dir(task):
    return f"_out-{task}"


def image_name():
    return "evaluation-harness"


def container_home():
    return "/app"


def container_output_dir():
    return "output"


def generations_path(model: str, task: str = ""):
    if task == "":
        return f"{container_output_dir()}/generations_{model}.json"
    else:
        return f"{container_output_dir()}/generations_{model}_{task}.json"


def exec_and_wait(cmd, work_dir=home_path(), env=os.environ.copy()):
    cwd = os.getcwd()
    os.chdir(work_dir)
    proc = subprocess.Popen([cmd], shell=True)
    proc.wait()
    os.chdir(cwd)


def docker_run_cmd(gpus: List[int], model: str, task: str):
    container_name = "cse710-project-eval"

    basic_flags = ["--shm-size=32G", "-it", "--rm", f"--name {container_name}"]

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


def run_LLM(
    gpus: List[int],
    model: str,
    pt: str,
    task: str,
    max_length_generation,
    temperature,
    do_sample,
    n_samples,
    batch_size,
):

    local_pt_path = f"{home_path()}/{models_dir()}/{model}/{pt}"
    if os.path.isfile(local_pt_path):
        pt_path = f"{container_home()}/{model}/{pt}"
    else:
        pt_path = "\"\""

    flags = [
        f"--model {container_home()}/{model}",
        f"--pt {pt_path}",
        f"--tasks {task}",
        f"--max_length_generation {max_length_generation}",
        f"--temperature {temperature}",
        f"--do_sample {do_sample}",
        f"--n_samples {n_samples}",
        f"--batch_size {batch_size}",
        "--trust_remote_code",
        "--generation_only",
        "--save_generations",
        f"--save_generations_path  {generations_path(model)}",
    ]

    gen_cmd = "accelerate launch main.py " + " ".join(flags)

    cmd = docker_run_cmd(gpus, model, task) + " " + gen_cmd
    print(cmd)
    exec_and_wait(cmd)


def eval_generations(gpus: List[int], model: str, task: str, temperature, n_samples):
    flags = [
        f"--model {container_home()}/{model}",
        f"--tasks {task}",
        f"--load_generations_path {generations_path(model, task)}",
        "--allow_code_execution",
        f"--temperature {temperature}",
        f"--n_samples {n_samples}",
    ]

    eval_cmd = (
        "python3 main.py "
        + " ".join(flags)
        + f" > {container_output_dir()}/eval_{model}_{task}.txt"
    )

    cmd = docker_run_cmd(gpus, model, task) + ' sh -c "' + eval_cmd + '"'
    print(cmd)
    exec_and_wait(cmd)


def main():
    start_time = time.time()

    args = parse_arg()
    if args.inspect:
        NOT_IMPLEMENTED()
    elif args.task in ["humaneval", "mercury"] or args.task.startswith("codexglue"):
        run_LLM(
            args.gpus,
            args.model,
            args.pt,
            args.task,
            args.max_length_generation,
            args.temperature,
            args.do_sample,
            args.n_samples,
            args.batch_size,
        )
        eval_generations(args.gpus, args.model, args.task, args.temperature, args.n_samples)
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
