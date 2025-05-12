import json
import logging
import os
from pathlib import Path
from typing import Tuple

import click
import typer
from typing_extensions import Annotated

from skythought.evals.common.entities import (
    Backend,
    BackendParameters,
    SamplingParameters,
)
from skythought.evals.inference_and_check import (
    generate_and_save,
    generate_and_score,
    score_results,
    score_responses
)
from skythought.evals.models import ModelConfig, get_system_prompt_keys
from skythought.evals.tasks import TASK_HANDLER_MAP, TASK_NAMES_TO_YAML, TaskConfig
from skythought.evals.util.cli_util import get_deterministic_hash, parse_multi_args
from skythought.evals.util.common import set_seed
from skythought.evals.util.results import SummaryResults, save_summary
from skythought.evals.util.metrics import pass_at_k

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = typer.Typer(pretty_exceptions_enable=False)


def get_run_config(
    task: str,
    task_config: TaskConfig,
    model_config: ModelConfig,
    backend: Backend,
    backend_args_as_dict: dict,
    sampling_params_as_dict: dict,
    start: int,
    end: int,
) -> dict:
    return {
        "task": {
            "name": task,
            "config": task_config.model_dump(),
            "start": start,
            "end": end,
        },
        "model": {
            "name": model_config.model_id,
            "config": model_config.model_dump(),
        },
        "backend": {
            "name": backend,
            "backend_args": backend_args_as_dict,
        },
        "sampling_params": sampling_params_as_dict,
    }


def parse_common_args(
    *,
    task: str,
    model: str,
    task_args: str,
    backend: Backend,
    backend_args: str,
    sampling_params: str,
    n: int,
    batch_size: int,
    system_prompt: str,
    assistant_prefill: str,
) -> Tuple[
    str,
    dict,
    str,
    Backend,
    dict,
    BackendParameters,
    SamplingParameters,
    dict,
    int,
    int,
    str,
]:
    # For strings passed via CLI, recover escape characters properly. This is hacky but works and convenient for short strings
    system_prompt = (
        system_prompt.encode("utf-8").decode("unicode_escape")
        if system_prompt
        else None
    )
    assistant_prefill = (
        assistant_prefill.encode("utf-8").decode("unicode_escape")
        if assistant_prefill
        else None
    )

    # TODO (sumanthrh): We should ideally read from ctx and get user-provided params
    if batch_size != 64 and backend not in (Backend.VLLM, Backend.HF_LOCAL):
        raise ValueError("Batch size is only supported for the vllm backend.")

    # Enable hf_transfer if not overridden by the user
    if os.environ.get("HF_HUB_ENABLE_HF_TRANSFER", None) is None:
        os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

    if task not in TASK_NAMES_TO_YAML:
        raise ValueError(
            f"Task {task} not found. Should be one of {TASK_NAMES_TO_YAML.keys()}"
        )
    task_args_as_dict = parse_multi_args(task_args)
    sampling_params_as_dict = parse_multi_args(sampling_params)
    backend_args_as_dict = parse_multi_args(backend_args)

    if n is not None:
        sampling_params_as_dict["n"] = n

    sampling_params: SamplingParameters = SamplingParameters.from_dict(
        backend, sampling_params_as_dict
    )
    backend_params: BackendParameters = BackendParameters.from_dict(
        backend, backend_args_as_dict
    )

    if sampling_params.params.top_p < 1 and model.startswith("openai/o1"):
        print(
            "OpenAI o1 models do not support `top_p` sampling. Resetting `top_p` to 1"
        )
        sampling_params.params.top_p = 1
        sampling_params_as_dict["top_p"] = 1

    if sampling_params.params.temperature == 0 and sampling_params.params.n > 1:
        sampling_params.params.n = 1
        sampling_params_as_dict["n"] = 1
        logger.warning(
            "Warning: Temperature 0 does not support multiple samples. Setting n=1."
        )

    return (
        task,
        task_args_as_dict,
        model,
        backend,
        backend_args_as_dict,
        backend_params,
        sampling_params_as_dict,
        sampling_params,
        n,
        batch_size,
        system_prompt,
        assistant_prefill,
    )


def get_output_dir(
    result_dir,
    *,
    model_id: str,
    task: str,
    start: int,
    end: int,
    run_config: dict,
) -> Path:
    parameter_hash = get_deterministic_hash(run_config)

    return Path(result_dir) / f"{model_id.replace('/', '_')}_{task}_{parameter_hash}"


@app.command("evaluate", help="Evaluate a model on a task")
def evaluate(
    ctx: typer.Context,
    task: Annotated[
        str,
        typer.Option(
            ...,
            help="Task to process.",
            click_type=click.Choice(list(TASK_NAMES_TO_YAML.keys())),
            case_sensitive=False,
        ),
    ],
    model: Annotated[str, typer.Option(..., help="The model to run")],
    backend: Annotated[
        Backend,
        typer.Option(
            help="Backend to use for inference.",
            case_sensitive=False,
        ),
    ] = Backend.VLLM,
    backend_args: Annotated[
        str,
        typer.Option(
            help="Backend parameters to use for inference. For open-source models, we perform inference in float32 by default",
            case_sensitive=False,
        ),
    ] = "",
    sampling_params: Annotated[
        str,
        typer.Option(
            help="Sampling parameters to use for inference.",
            case_sensitive=False,
        ),
    ] = "temperature=0,top_p=1,max_tokens=32768",
    result_dir: Annotated[
        str,
        typer.Option(
            help="Result directory to save outputs.",
        ),
    ] = "./",
    system_prompt_name: Annotated[
        str,
        typer.Option(
            help="System prompt template to use, overriding any pre-configured system prompt for this model.",
            click_type=click.Choice(get_system_prompt_keys()),
        ),
    ] = None,
    system_prompt: Annotated[
        str,
        typer.Option(
            help="System prompt to use, overriding any pre-configured system prompt for this model."
        ),
    ] = None,
    n: Annotated[
        int, typer.Option(help="Number of samples generated per problem.")
    ] = None,
    seed: Annotated[int, typer.Option(help="Random seed.")] = 41,
    assistant_prefill: Annotated[
        str,
        typer.Option(help=r'Assistant prefill for the model response. Ex: "<think>\n"'),
    ] = None,
    as_test: Annotated[
        bool, typer.Option(help="Perform a test run on 10 samples of the dataset.")
    ] = False,
    overwrite: Annotated[
        bool, typer.Option(help="Overwrite existing results.")
    ] = False,
    batch_size: Annotated[
        int,
        typer.Option(
            help="Batch size for inference. only applicable for the vllm backend."
        ),
    ] = 64,
):
    set_seed(seed)

    (
        task,
        _,
        model,
        backend,
        backend_args_as_dict,
        backend_params,
        sampling_params_as_dict,
        sampling_params,
        n,
        batch_size,
        system_prompt,
        assistant_prefill,
    ) = parse_common_args(
        task=task,
        model=model,
        # `evaluate` does not allow customization of `task_args`
        task_args="",
        backend=backend,
        backend_args=backend_args,
        sampling_params=sampling_params,
        n=n,
        batch_size=batch_size,
        system_prompt=system_prompt,
        assistant_prefill=assistant_prefill,
    )
    # ensure parsing was correct
    assert isinstance(sampling_params, SamplingParameters)
    logger.info(
        f"Temperature: {sampling_params.params.temperature}, top_p: {sampling_params.params.top_p}, max_tokens: {sampling_params.params.max_tokens}"
    )

    start = 0
    end = -1
    if as_test:
        start = 0
        end = 10
        sampling_params.params.max_tokens = 2048
        logger.info("Running test run with 10 samples and max tokens set to 2048.")

    task_config = TaskConfig.from_yaml(TASK_NAMES_TO_YAML[task])
    handler_name = task_config.handler
    handler_cls = TASK_HANDLER_MAP[handler_name]
    handler = handler_cls(task_config)

    model_config = ModelConfig.from_model_id(
        model, system_prompt_name, system_prompt, assistant_prefill
    )
    
    run_config_dict = get_run_config(
        task,
        task_config,
        model_config,
        backend,
        backend_args_as_dict,
        sampling_params_as_dict,
        start,
        end,
    )

    output_dir = get_output_dir(
        result_dir,
        model_id=model,
        task=task,
        start=start,
        end=end,
        run_config=run_config_dict,
    )
    if not overwrite and output_dir.exists() and len(os.listdir(output_dir)) != 0:
        raise ValueError(
            f"Output directory {output_dir} already exists. pass `--overwrite` to overwrite."
        )
    # create result dir if not exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    generate_and_score(
        handler,
        model_config,
        backend,
        backend_params,
        sampling_params,
        output_dir,
        start,
        end,
        run_config_dict,
        batch_size=batch_size,
    )


@app.command("generate", help="Generate model response for a task and save results")
def generate(
    task: Annotated[
        str,
        typer.Option(
            ...,
            help="Task to process.",
            click_type=click.Choice(list(TASK_NAMES_TO_YAML.keys())),
            case_sensitive=False,
        ),
    ],
    model: Annotated[str, typer.Option(..., help="The model to run")],
    task_args: Annotated[
        str,
        typer.Option(
            help="Task arguments to use for inference.",
        ),
    ] = "",
    backend: Annotated[
        Backend,
        typer.Option(
            help="Backend to use for inference.",
            case_sensitive=False,
        ),
    ] = Backend.VLLM,
    backend_args: Annotated[
        str,
        typer.Option(
            help="Backend parameters to use for inference.",
            case_sensitive=False,
        ),
    ] = "",
    sampling_params: Annotated[
        str,
        typer.Option(
            help="Sampling parameters to use for inference.",
            case_sensitive=False,
        ),
    ] = "temperature=0,top_p=1,max_tokens=32768",
    result_dir: Annotated[
        str,
        typer.Option(
            help="Result directory to save outputs.",
        ),
    ] = None,
    system_prompt_name: Annotated[
        str,
        typer.Option(
            help="System prompt template to use, overriding any pre-configured system prompt for this model.",
            click_type=click.Choice(get_system_prompt_keys()),
        ),
    ] = None,
    system_prompt: Annotated[
        str,
        typer.Option(
            help="System prompt to use, overriding any pre-configured system prompt for this model."
        ),
    ] = None,
    n: Annotated[
        int, typer.Option(help="Number of samples generated per problem.")
    ] = None,
    seed: Annotated[int, typer.Option(help="Random seed.")] = 41,
    assistant_prefill: Annotated[
        str,
        typer.Option(help=r'Assistant prefill for the model response. Ex: "<think>\n"'),
    ] = None,
    overwrite: Annotated[
        bool, typer.Option(help="Overwrite existing results.")
    ] = False,
    batch_size: Annotated[
        int,
        typer.Option(
            help="Batch size for inference. only applicable for the vllm backend."
        ),
    ] = 64,
    start: Annotated[int, typer.Option(help="Start index for the dataset.")] = 0,
    end: Annotated[
        int,
        typer.Option(
            help="End index for the dataset (non-inclusive). If a negative value is provided, we use all the samples."
        ),
    ] = -1,
    resume_from: Annotated[
        str, typer.Option(help="Resume from a previous run.")
    ] = None,
):
    set_seed(seed)

    (
        task,
        task_args_as_dict,
        model,
        backend,
        backend_args_as_dict,
        backend_params,
        sampling_params_as_dict,
        sampling_params,
        n,
        batch_size,
        system_prompt,
        assistant_prefill,
    ) = parse_common_args(
        task=task,
        model=model,
        task_args=task_args,
        backend=backend,
        backend_args=backend_args,
        sampling_params=sampling_params,
        n=n,
        batch_size=batch_size,
        system_prompt=system_prompt,
        assistant_prefill=assistant_prefill,
    )

    assert isinstance(sampling_params, SamplingParameters)
    logger.info(
        f"Temperature: {sampling_params.params.temperature}, top_p: {sampling_params.params.top_p}, max_tokens: {sampling_params.params.max_tokens}"
    )

    task_config = TaskConfig.from_yaml(TASK_NAMES_TO_YAML[task])
    # update with user-provided args
    task_config.update(**task_args_as_dict)

    handler_name = task_config.handler
    handler_cls = TASK_HANDLER_MAP[handler_name]
    handler = handler_cls(task_config)

    model_config = ModelConfig.from_model_id(
        model, system_prompt_name, system_prompt, assistant_prefill
    )

    output_dir = None
    if resume_from is not None:
        resume_from = Path(resume_from)
        if not resume_from.exists():
            raise ValueError(f"Output directory {resume_from} does not exist.")

    assert (resume_from is None) ^ (
        result_dir is None
    ), "One of `resume_from` or `result_dir` must be true."

    run_config_dict = get_run_config(
        task,
        task_config,
        model_config,
        backend,
        backend_args_as_dict,
        sampling_params_as_dict,
        start,
        end,
    )

    if result_dir is not None:
        output_dir = get_output_dir(
            result_dir,
            model_id=model,
            task=task,
            start=start,
            end=end,
            run_config=run_config_dict,
        )
        if not overwrite and output_dir.exists() and len(os.listdir(output_dir)) != 0:
            raise ValueError(
                f"Output directory {output_dir} already exists. pass `--overwrite` to overwrite."
            )
        # create result dir if not exists
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    generate_and_save(
        handler,
        model_config,
        backend,
        backend_params,
        sampling_params,
        output_dir,
        start,
        end,
        run_config_dict,
        resume_from=resume_from,
        batch_size=batch_size,
    )

import json
import pandas as pd

import json
import pandas as pd

def generate_id_to_results(results_file: str) -> dict:
    """
    从 results.json 文件生成 id_to_results 字典。

    参数:
        results_file (str): results.json 文件的路径。

    返回:
        dict: 一个字典，其中键是唯一 ID，值是包含来自 results.json 文件的数据的字典。
              如果文件不存在或其他错误，则返回一个空字典。
    """
    id_to_results = {}
    try:
        with open(results_file, "r") as f:
            data = json.load(f)  # 加载整个 JSON 文件
            if not isinstance(data, dict): # 修改了这里，以适应给定的JSON 结构
                print(f"Error: Expected a dict of results, but got {type(data)}.  Returning empty dict.")
                return {}

            for key, item in data.items(): # 遍历字典的键值对
                # 检查 item 是否为字典
                if not isinstance(item, dict):
                    print(f"Error: Expected each item in the dict to be a dictionary, but got {type(item)}. Skipping this item.")
                    continue
                # 检查唯一 ID 是否存在。现在直接使用字典的键
                unique_id = str(key)  # 确保 ID 是字符串类型

                # 存储整个 item。我们将像原始代码一样向其添加更多内容。
                id_to_results[unique_id] = item

                # 原始代码添加了这些键。如果它们不存在，则将它们添加为 None 或 []
                if "responses" not in item:
                    id_to_results[unique_id]["responses"] = []
                if "token_usages" not in item:
                    id_to_results[unique_id]["token_usages"] = []
                if "prompt" not in item:
                    id_to_results[unique_id]["prompt"] = None
                if "input_conversation" not in item:
                    id_to_results[unique_id]["input_conversation"] = None
                if "activation_file" not in item:
                    id_to_results[unique_id]["activation_file"] = None

    except FileNotFoundError:
        print(f"Error: File not found at {results_file}. Returning empty dict.")
        return {}
    except json.JSONDecodeError as e:
        print(f"Error: JSONDecodeError: {e}.  Invalid JSON in {results_file}. Returning empty dict.")
        return {}
    except Exception as e:
        print(f"An unexpected error occurred: {e}. Returning empty dict.")
        return {}
    return id_to_results



@app.command("score", help="Score a model on a task")
def score(
    run_dir: Annotated[
        str, typer.Option(..., help="The directory of saved results to score")
    ],
    task: Annotated[
        str,
        typer.Option(
            ...,
            help="Task to process.",
            click_type=click.Choice(list(TASK_NAMES_TO_YAML.keys())),
            case_sensitive=False,
        ),
    ],
):
    if not os.path.exists(run_dir):
        raise ValueError(f"Run directory {run_dir} does not exist.")

    run_dir = Path(run_dir)

    if task not in TASK_NAMES_TO_YAML:
        raise ValueError(
            f"Task {task} not found. Should be one of {TASK_NAMES_TO_YAML.keys()}"
        )

    task_config = TaskConfig.from_yaml(TASK_NAMES_TO_YAML[task])
    handler_name = task_config.handler
    handler_cls = TASK_HANDLER_MAP[handler_name]
    handler = handler_cls(task_config)

    id_to_results = generate_id_to_results(run_dir / "results.json")
    accuracy, id_to_scores, total_finish = score_responses(
        handler, id_to_results, max_workers=8
    )
    logger.info(f"Accuracy: {accuracy}")

    sample_n = len(next(iter(id_to_results.values()))['responses'])
    num_responses_total = len(id_to_results) * sample_n

    pass_at_k_metrics = None
    if sample_n > 1:
        pass_at_k_metrics = pass_at_k(sample_n, id_to_scores)

    total_completion_tokens = None
    total_prompt_tokens = None
    summary_data = SummaryResults(
        configuration=None,
        total_completion_tokens=total_completion_tokens,
        total_prompt_tokens=total_prompt_tokens,
        avg_completion_tokens=(
            round(total_completion_tokens / num_responses_total, 3)
            if total_completion_tokens
            else 0
        ),
        avg_prompt_tokens=(
            round(total_prompt_tokens / num_responses_total, 3)
            if total_prompt_tokens
            else 0
        ),
        accuracy=accuracy,
        pass_at_k=pass_at_k_metrics,
    )

    summary_file = run_dir / "summary.json"
    save_summary(summary_file, summary_data)

    # get run_config from run_dir
    # summary_file = run_dir / "summary.json"
    # if not summary_file.exists():
    #     raise ValueError(f"Run summary file {summary_file} does not exist.")

    # with open(summary_file, "r") as f:
    #     run_summary = json.load(f)

    # run_summary = SummaryResults(**run_summary)

    # score_results(handler, run_dir, run_summary)


def main():
    app()

if __name__ == "__main__":
    root_dirs = [
        'skythought/evaluate_results/temp0.6-tp95/math-long-cot-40k-32768',
        'skythought/evaluate_results/temp0.6-tp95/math-long-cot-80k-32768',
        'skythought/evaluate_results/temp0.6-tp95/math-long-cot-80k-16384',
    ]
    for root_dir in root_dirs:
        for root, dirs, files in os.walk(root_dir):
            if 'results.json' in files and 'summary.json' not in files:
                path_parts = root.split(os.sep)
                dataset = path_parts[-2]  # 父目录是数据集名称
                model_dir = path_parts[-1]  # 当前目录是模型目录
                print(dataset, model_dir)
                print(root, '\n\n')
                score(root, dataset)