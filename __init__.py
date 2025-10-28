from __future__ import annotations
from typing import Union, Any, Tuple, Optional
import time
import datetime
import logging
import os
import sys
import json
import argparse
import yaml
import psutil

from comfy_api.latest import ComfyExtension, io
import execution
import folder_paths
import comfy.samplers
import comfy.model_patcher
import comfy.patcher_extension
import comfy.model_management
import comfy.cli_args
import comfy.utils
import comfy.sd


if comfy.model_management.is_nvidia():
    from threading import Thread
    from queue import Queue, Empty
    import subprocess
from .nodes import BenchmarkWorkflow  # Import the custom node


VERSION = 0
# currently, for simplicity we use a global variable to store the execution context;
# this allows us to access the execution context from all hooks without having to remake the hooks at runtime
GLOBAL_CONTEXT = None
ENABLE_NVIDIA_SMI_DATA = False
INITIAL_NVIDIA_SMI_QUERY = None
INFO_NVIDIA_SMI_QUERY = None
NVIDIA_SMI_ERROR = None

INITIAL_PSUTIL_QUERY = None
psutil_query = ["timestamp", "used", "total"]

nvidia_smi_query = ["timestamp", "memory.used", "memory.total", "utilization.gpu", "utilization.memory", "power.draw", "power.draw.instant", "power.limit", "pcie.link.gen.current", "pcie.link.gen.max", "pcie.link.width.current"]
_nvidia_smi_query_list = ["nvidia-smi", "--query-gpu=" + ",".join(nvidia_smi_query), "--format=csv,noheader,nounits"]

info_nvidia_smi_query = ["name", "count", "driver_version", "display_active", "vbios_version", "power.management"]
_info_nvidia_smi_query_list = ["nvidia-smi", "--query-gpu=" + ",".join(info_nvidia_smi_query), "--format=csv,noheader,nounits"]

# For NVIDIA devices, during the benchmark setup a process to call nvidia-smi regularly (or with varying intervals)
def nvidia_smi_thread(out_queue: Queue, in_queue: Queue, check_interval: float):
    logging.info("Starting nvidia-smi thread")
    while True:
        try:
            out_queue.put(call_nvidia_smi(_nvidia_smi_query_list, raise_error=True))
            out_queue.put(call_psutil())
        except Exception as e:
            logging.error(f"Breaking out of nvidia-smi thread due to {e}")
            break
        try:
            item = in_queue.get(timeout=check_interval)
            if item == "stop":
                break
        except Empty:
            pass
        except Exception as e:
            logging.error(f"Breaking out of nvidia-smi thread reading in_queue due to {e}")
            break
    logging.info("Exiting nvidia-smi thread")

def create_nvidia_smi_thread(check_interval: float):
    out_queue = Queue()
    in_queue = Queue()
    thread = Thread(target=nvidia_smi_thread, args=(out_queue, in_queue, check_interval))
    thread.daemon = True
    thread.start()
    return out_queue, in_queue, thread

def get_from_nvidia_smi(query: str, param: str, _query_list: list[str]=nvidia_smi_query):
    if param not in nvidia_smi_query:
        return None
    index = nvidia_smi_query.index(param)
    values = query.split(",")
    return values[index]


def json_func(obj):
    try:
        return str(obj)
    except Exception:
        return "Error converting to json"

def get_provided_args(args, parser):
    """
    Return only the arguments that were explicitly provided
    (i.e. differ from their defaults).
    """
    defaults = {a.dest: a.default for a in parser._actions if a.dest != argparse.SUPPRESS}
    args_dict = vars(args)
    return {k: v for k, v in args_dict.items() if defaults.get(k) != v}

# NOTE: the version of this in comfy.patcher_extension has internal dict merge wrong; so use this instead for now
def merge_nested_dicts(dict1: dict, dict2: dict, copy_dict1=True):
    if copy_dict1:
        merged_dict = comfy.patcher_extension.copy_nested_dicts(dict1)
    else:
        merged_dict = dict1
    for key, value in dict2.items():
        if isinstance(value, dict):
            curr_value = merged_dict.setdefault(key, {})
            merged_dict[key] = merge_nested_dicts(curr_value, value)
        elif isinstance(value, list):
            merged_dict.setdefault(key, []).extend(value)
        else:
            merged_dict[key] = value
    return merged_dict

def load_config():
    default_config = {
            "nvidia-smi": {
                "enable_thread": True,
                "check_interval": 0.25
            },
            "log": {
                "iteration_times": True,
            },
            "options": {
                "require_node": False,
            },
            "name_prefix": "benchmark"
        }
    config_file = os.path.join(os.path.dirname(__file__), "config.yaml")
    if os.path.exists(config_file):
        try:
            with open(config_file, "r") as f:
                loaded_config = yaml.safe_load(f)
                config = merge_nested_dicts(default_config, loaded_config, copy_dict1=True)
        except yaml.reader.ReaderError as e:
            logging.error(f"ReaderError loading config.yaml for comfyui-benchmark, will replace with default config.json instead: {e}")
            config = default_config
        with open(config_file, "w") as f:
            yaml.dump(config, f)
    else:
        logging.info(f"No config.yaml found for comfyui-benchmark, creating default config at {config_file}")
        config = default_config
        with open(config_file, "w") as f:
            yaml.dump(config, f)
    return config

class ExecutionContext:
    def __init__(self):
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        config = load_config()
        self.workflow_name = f"{config.get('name_prefix', '')}_{timestamp}"
        self.version = VERSION
        self.device_info = {
            "name": comfy.model_management.get_torch_device_name(comfy.model_management.get_torch_device()),
            "pytorch_version": comfy.model_management.torch_version,
            "operating_system": str(sys.platform),
            "vram_state": comfy.model_management.vram_state.name,
            "total_vram": comfy.model_management.total_vram,
            "total_ram": comfy.model_management.total_ram,
        }
        self.benchmark_data: dict[str] = {}
        self.load_data: dict[str,list[dict[str]]] = {
            "load_diffusion_model": [],
            "load_torch_file": [],
            "load_state_dict": [],
            "model_load": [],
            "model_unload": [],
            "patch_model": [],
            "unpatch_model": [],
        }
        self.sampling_data: list[dict[str]] = []
        self.vae_data: dict[str,list[dict[str]]] = {
            "encode": [],
            "decode": [],
        }
        self.clip_data: dict[str,list[dict[str]]] = {
            "encode": [],
            "tokenize": [],
        }
        self.caches_data: dict[str,list[dict[str]]] = {
            "clean_unused": [],
        }
        self.startup_args = get_provided_args(comfy.cli_args.args, comfy.cli_args.parser)
        self.config: dict[str, Union[dict[str], Any]] = config
        self.nvidia_smi_data_info: dict[str, str] = {}
        self.nvidia_smi_data: list[str] = []
        self.psutil_data_info: dict[str, str] = {}
        self.psutil_data: list[str] = []
        self._create_nvidia_smi_data_info()
        self._create_psutil_data_info()

    def is_nvidia_smi_thread_enabled(self):
        return self.config.get("nvidia-smi", {}).get("enable_thread", False)

    def get_nvidia_smi_check_interval(self):
        return self.config.get("nvidia-smi", {}).get("check_interval", 0.25)

    def get_log_dict(self):
        return self.config.get("log", {})

    def is_require_node_enabled(self):
        return self.config.get("options", {}).get("require_node", False)

    def _create_nvidia_smi_data_info(self):
        self.nvidia_smi_data_info["info_nvidia_smi_query_params"] = ", ".join(info_nvidia_smi_query)
        self.nvidia_smi_data_info["info_nvidia_smi_query"] = INFO_NVIDIA_SMI_QUERY
        self.nvidia_smi_data_info["nvidia_smi_query_params"] = ", ".join(nvidia_smi_query)
        self.nvidia_smi_data_info["initial_nvidia_smi_query"] = INITIAL_NVIDIA_SMI_QUERY
        self.nvidia_smi_data_info["nvidia_smi_error"] = NVIDIA_SMI_ERROR

    def _create_psutil_data_info(self):
        self.psutil_data_info["psutil_query_params"] = ", ".join(psutil_query)
        self.psutil_data_info["initial_psutil_query"] = INITIAL_PSUTIL_QUERY

    def save_to_log_file(self, prompt: dict = None):
        output_dir = folder_paths.get_output_directory()
        benchmark_dir = os.path.join(output_dir, "benchmark")
        os.makedirs(benchmark_dir, exist_ok=True)
        # Check for BenchmarkWorkflow node and get postfixes
        prefix = ""
        postfix = ""
        if prompt is not None:
            has_benchmark_node, prefix, postfix = check_workflow_for_benchmark_node(prompt)
            if prefix:
                prefix = f"{prefix}_"
            if postfix:
                postfix = f"_{postfix}"
        benchmark_name = f"{prefix}{self.workflow_name}{postfix}"
        benchmark_file = os.path.join(benchmark_dir, f"{benchmark_name}.json")
        try:
            with open(benchmark_file, "w") as f:
                json.dump(self.__dict__, f, indent=4, ensure_ascii=False, default=json_func)
            logging.info(f"Benchmark: {benchmark_name} saved to {benchmark_file}")
        except Exception as e:
            logging.error(f"Error saving benchmark file {benchmark_file}: {e}")

def hook_CLIP():
    def factory_CLIP(func, func_name: str, category: str, check_nested=True):
        def wrapper_CLIP(*args, **kwargs):
            # don't track anything if already nested within a benchmark hook
            if check_nested and getattr(args[0], "_inside_benchmark_hook", False):
                return func(*args, **kwargs)
            global GLOBAL_CONTEXT
            context = GLOBAL_CONTEXT
            valid_timing = True
            try:
                if check_nested:
                    setattr(args[0], "_inside_benchmark_hook", True)
                start_time = time.perf_counter()
                return func(*args, **kwargs)
            except Exception as _:
                valid_timing = False
                raise
            finally:
                end_time = time.perf_counter()
                if check_nested and hasattr(args[0], "_inside_benchmark_hook"):
                    delattr(args[0], "_inside_benchmark_hook")
                if context is not None:
                    context.clip_data[category].append({
                        "model": str(args[0].cond_stage_model.__class__.__name__),
                        "func_name": func_name,
                        "elapsed_time": end_time - start_time,
                        "start_time": start_time,
                        "valid_timing": valid_timing,
                    })
        return wrapper_CLIP
    comfy.sd.CLIP.encode = factory_CLIP(comfy.sd.CLIP.encode, "CLIP.encode", "encode")
    comfy.sd.CLIP.encode_from_tokens = factory_CLIP(comfy.sd.CLIP.encode_from_tokens, "CLIP.encode_from_tokens", "encode")
    comfy.sd.CLIP.encode_from_tokens_scheduled = factory_CLIP(comfy.sd.CLIP.encode_from_tokens_scheduled, "CLIP.encode_from_tokens_scheduled", "encode")
    comfy.sd.CLIP.tokenize = factory_CLIP(comfy.sd.CLIP.tokenize, "CLIP.tokenize", "tokenize", check_nested=False)

def hook_VAE():
    def hook_VAE_encode():
        def factory_VAE_encode(func):
            def wrapper_VAE_encode(*args, **kwargs):
                global GLOBAL_CONTEXT
                context = GLOBAL_CONTEXT
                valid_timing = True
                try:
                    start_time = time.perf_counter()
                    return func(*args, **kwargs)
                except Exception as _:
                    valid_timing = False
                    raise
                finally:
                    end_time = time.perf_counter()
                    if context is not None:
                        context.vae_data["encode"].append({
                            "elapsed_time": end_time - start_time,
                            "start_time": start_time,
                            "valid_timing": valid_timing
                        })
            return wrapper_VAE_encode
        comfy.sd.VAE.encode = factory_VAE_encode(comfy.sd.VAE.encode)

    def hook_VAE_decode():
        def factory_VAE_decode(func):
            def wrapper_VAE_decode(*args, **kwargs):
                global GLOBAL_CONTEXT
                context = GLOBAL_CONTEXT
                valid_timing = True
                try:
                    start_time = time.perf_counter()
                    return func(*args, **kwargs)
                except Exception as _:
                    valid_timing = False
                    raise
                finally:
                    end_time = time.perf_counter()
                    if context is not None:
                        context.vae_data["decode"].append({
                            "elapsed_time": end_time - start_time,
                            "start_time": start_time,
                            "valid_timing": valid_timing
                        })
            return wrapper_VAE_decode
        comfy.sd.VAE.decode = factory_VAE_decode(comfy.sd.VAE.decode)
    hook_VAE_encode()
    hook_VAE_decode()

def hook_LoadedModel_model_load():
    def factory_LoadedModel_model_load(func, category: str, model_nest: int=2):
        def wrapper_LoadedModel_model_load(*args, **kwargs):
            global GLOBAL_CONTEXT
            context = GLOBAL_CONTEXT
            valid_timing = True
            try:
                start_time = time.perf_counter()
                return func(*args, **kwargs)
            except Exception as _:
                valid_timing = False
                raise
            finally:
                end_time = time.perf_counter()
                if context is not None:
                    context.load_data[category].append({
                        "model": str(args[0].model.model.__class__.__name__ if model_nest == 2 else args[0].model.__class__.__name__),
                        "elapsed_time": end_time - start_time,
                        "start_time": start_time,
                        "valid_timing": valid_timing
                    })
        return wrapper_LoadedModel_model_load
    comfy.model_management.LoadedModel.model_load = factory_LoadedModel_model_load(comfy.model_management.LoadedModel.model_load, "model_load")
    comfy.model_management.LoadedModel.model_unload = factory_LoadedModel_model_load(comfy.model_management.LoadedModel.model_unload, "model_unload")
    comfy.model_patcher.ModelPatcher.patch_model = factory_LoadedModel_model_load(comfy.model_patcher.ModelPatcher.patch_model, "patch_model", model_nest=1)
    comfy.model_patcher.ModelPatcher.unpatch_model = factory_LoadedModel_model_load(comfy.model_patcher.ModelPatcher.unpatch_model, "unpatch_model", model_nest=1)

def hook_load_state_dict():
    def factory_load_state_dict(func, func_name: str):
        def wrapper_load_state_dict(*args, **kwargs):
            global GLOBAL_CONTEXT
            context = GLOBAL_CONTEXT
            valid_timing = True
            try:
                start_time = time.perf_counter()
                return func(*args, **kwargs)
            except Exception as _:
                valid_timing = False
                raise
            finally:
                end_time = time.perf_counter()
                if context is not None:
                    context.load_data["load_state_dict"].append({
                        "func_name": func_name,
                        "elapsed_time": end_time - start_time,
                        "start_time": start_time,
                        "valid_timing": valid_timing
                    })
        return wrapper_load_state_dict

    def factory_load_diffusion_model(func, func_name: str):
        def wrapper_load_diffusion_model(*args, **kwargs):
            global GLOBAL_CONTEXT
            context = GLOBAL_CONTEXT
            valid_timing = True
            try:
                start_time = time.perf_counter()
                return func(*args, **kwargs)
            except Exception as _:
                valid_timing = False
                raise
            finally:
                end_time = time.perf_counter()
                if context is not None:
                    context.load_data["load_diffusion_model"].append({
                        "func_name": func_name,
                        "elapsed_time": end_time - start_time,
                        "start_time": start_time,
                        "valid_timing": valid_timing
                    })
        return wrapper_load_diffusion_model

    comfy.sd.load_state_dict_guess_config = factory_load_state_dict(comfy.sd.load_state_dict_guess_config, "load_state_dict_guess_config")
    comfy.sd.load_diffusion_model_state_dict = factory_load_state_dict(comfy.sd.load_diffusion_model_state_dict, "load_diffusion_model_state_dict")
    comfy.sd.load_text_encoder_state_dicts = factory_load_state_dict(comfy.sd.load_text_encoder_state_dicts, "load_text_encoder_state_dicts")
    comfy.sd.load_diffusion_model = factory_load_diffusion_model(comfy.sd.load_diffusion_model, "load_diffusion_model")
    comfy.sd.load_checkpoint_guess_config = factory_load_diffusion_model(comfy.sd.load_checkpoint_guess_config, "load_checkpoint_guess_config")
    comfy.sd.load_clip = factory_load_diffusion_model(comfy.sd.load_clip, "load_clip")

def hook_load_torch_file():
    def factory_load_torch_file(func):
        def wrapper_load_torch_file(*args, **kwargs):
            global GLOBAL_CONTEXT
            context = GLOBAL_CONTEXT
            valid_timing = True
            try:
                start_time = time.perf_counter()
                return func(*args, **kwargs)
            except Exception as _:
                valid_timing = False
                raise
            finally:
                end_time = time.perf_counter()
                if context is not None:
                    context.load_data["load_torch_file"].append({
                        "ckpt": args[0],
                        "elapsed_time": end_time - start_time,
                        "start_time": start_time,
                        "valid_timing": valid_timing
                    })
        return wrapper_load_torch_file
    comfy.utils.load_torch_file = factory_load_torch_file(comfy.utils.load_torch_file)

def hook_CFGGuider_sample():
    def add_predict_noise_wrapper(model_options: dict, context: ExecutionContext, temp_dict: dict[str]):
        def factory_predict_noise(c, temp_dict: dict[str]):
            def wrapper_predict_noise(executor, *args, **kwargs):
                temp_dict.setdefault("iteration_times", [])
                try:
                    start_time = time.perf_counter()
                    return executor(*args, **kwargs)
                finally:
                    end_time = time.perf_counter()
                    temp_dict["iteration_times"].append(end_time - start_time)
            return wrapper_predict_noise
        comfy.patcher_extension.add_wrapper_with_key(comfy.patcher_extension.WrappersMP.PREDICT_NOISE, "benchmark_sampling", factory_predict_noise(context, temp_dict),
                                            model_options, is_model_options=True)

    def add_sampler_sample_wrapper(model_options: dict, context: ExecutionContext, temp_dict: dict[str]):
        def factory_sampler_sample(c, temp_dict: dict[str]):
            def wrapper_sampler_sample(executor,*args, **kwargs):
                try:
                    start_time = time.perf_counter()
                    return executor(*args, **kwargs)
                finally:
                    end_time = time.perf_counter()
                    temp_dict["sampler_sample_elapsed_time"] = end_time - start_time
                    temp_dict["sampler_sample_start_time"] = start_time
                    temp_dict["sampler_sample_end_time"] = end_time
            return wrapper_sampler_sample
        comfy.patcher_extension.add_wrapper_with_key(comfy.patcher_extension.WrappersMP.SAMPLER_SAMPLE, "benchmark_sampling", factory_sampler_sample(context, temp_dict),
                                            model_options, is_model_options=True)

    def factory_CFGGuider_sample(func):
        def wrapper_CFGGuider_sample(*args, **kwargs):
            global GLOBAL_CONTEXT
            args = args
            kwargs = kwargs
            try:
                guider = args[0]
                orig_model_options = guider.model_options
                model_options = comfy.model_patcher.create_model_options_clone(orig_model_options)
                temp_dict = {}
                temp_dict["model"] = guider.model_patcher.model.__class__.__name__
                temp_dict["steps"] = len(args[4])-1  # NOTE: uses sigmas passed into sample() function
                if GLOBAL_CONTEXT is not None:
                    if GLOBAL_CONTEXT.get_log_dict().get("iteration_times", True):
                        add_predict_noise_wrapper(model_options, GLOBAL_CONTEXT, temp_dict)
                    add_sampler_sample_wrapper(model_options, GLOBAL_CONTEXT, temp_dict)
                guider.model_options = model_options
                cfg_guider_start_time = time.perf_counter()
                return func(*args, **kwargs)
            finally:
                cfg_guider_end_time = time.perf_counter()
                temp_dict["cfg_guider_elapsed_time"] = cfg_guider_end_time - cfg_guider_start_time
                temp_dict["cfg_guider_start_time"] = cfg_guider_start_time
                temp_dict["cfg_guider_end_time"] = cfg_guider_end_time
                if "iteration_times" in temp_dict:
                    temp_dict["average_iteration_time"] = sum(temp_dict["iteration_times"]) / len(temp_dict["iteration_times"])
                else:
                    temp_dict["average_iteration_time"] = -1
                if GLOBAL_CONTEXT is not None:
                    GLOBAL_CONTEXT.sampling_data.append(temp_dict)
                guider.model_options = orig_model_options
        return wrapper_CFGGuider_sample
    comfy.samplers.CFGGuider.sample = factory_CFGGuider_sample(comfy.samplers.CFGGuider.sample)

def hook_PromptExecutor_caches_clean_unused(executor: execution.PromptExecutor):
    def factory_cache_clean_unused(func, cache_name: str):
        def wrapper_cache_clean_unused(*args, **kwargs):
            global GLOBAL_CONTEXT
            context = GLOBAL_CONTEXT
            start_time = time.perf_counter()
            try:
                return func(*args, **kwargs)
            finally:
                end_time = time.perf_counter()
                if context is not None:
                    context.caches_data["clean_unused"].append({
                        "cache_name": cache_name,
                        "elapsed_time": end_time - start_time,
                        "start_time": start_time,
                    })
        return wrapper_cache_clean_unused
    if not hasattr(executor, "_hooked_by_benchmark"):
        executor.caches.outputs.clean_unused = factory_cache_clean_unused(executor.caches.outputs.clean_unused, f"outputs:{executor.caches.outputs.__class__.__name__}")
        executor.caches.ui.clean_unused = factory_cache_clean_unused(executor.caches.ui.clean_unused, f"ui:{executor.caches.ui.__class__.__name__}")
        executor.caches.objects.clean_unused = factory_cache_clean_unused(executor.caches.objects.clean_unused, f"objects:{executor.caches.objects.__class__.__name__}")
        setattr(executor, "_hooked_by_benchmark", True)

def check_workflow_for_benchmark_node(prompt: dict) -> Tuple[bool, str, str]:
    """
    Check if the workflow contains a BenchmarkWorkflow node with capture_benchmark=True.
    Returns a tuple of (has_benchmark_node, file_prefix, file_postfix).
    """
    for node_id, node in prompt.items():
        if not isinstance(node, dict):
            continue
        class_type = node.get("class_type")
        inputs = node.get("inputs", {})
        if class_type == "BenchmarkWorkflow" and inputs.get("capture_benchmark", False):
            return True, str(inputs.get("file_prefix", "")), str(inputs.get("file_postfix", ""))
    return False, "", ""

def hook_PromptExecutor_execute():
    def factory_PromptExecutor_execute(func):
        '''
        Create wrapper function that will time the total execution time for a workflow.
        '''
        def wrapper_PromptExecutor_execute(executor, prompt, *args, **kwargs):
            global GLOBAL_CONTEXT, ENABLE_NVIDIA_SMI_DATA, INITIAL_NVIDIA_SMI_QUERY, INFO_NVIDIA_SMI_QUERY, NVIDIA_SMI_ERROR
            args = args
            kwargs = kwargs
            # create execution context
            context = ExecutionContext()
            GLOBAL_CONTEXT = context
            # if its an nvidia card, we can do overall memory usage tracking via nvidia-smi calls
            should_save_benchmark = True
            if context.is_require_node_enabled():
                should_save_benchmark = check_workflow_for_benchmark_node(prompt)[0]
            thread_started = False
            if should_save_benchmark and context.is_nvidia_smi_thread_enabled() and ENABLE_NVIDIA_SMI_DATA:
                out_queue, in_queue, thread = create_nvidia_smi_thread(context.get_nvidia_smi_check_interval())
                thread_started = True
            # hook caches, but only once per startup of ComfyUI
            hook_PromptExecutor_caches_clean_unused(executor)
            try:
                start_datetime = datetime.datetime.now().strftime("%Y/%m/%d %H:%M:%S.%f")
                start_time = time.perf_counter()
                result = func(executor, prompt, *args, **kwargs)
                end_time = time.perf_counter()
                context.benchmark_data["execution_elapsed_time"] = end_time - start_time
                context.benchmark_data["workflow_start_datetime"] = start_datetime
                context.benchmark_data["workflow_start_time"] = start_time
                context.benchmark_data["workflow_end_time"] = end_time
                return result
            finally:
                if thread_started:
                    try:
                        in_queue.put("stop")
                        thread.join()
                        while not out_queue.empty():
                            item: str = out_queue.get_nowait()
                            if item.startswith("psutil:"):
                                context.psutil_data.append(item.split(":")[1])
                            else:
                                context.nvidia_smi_data.append(item)
                    except Exception as e:
                        logging.error(f"Error stopping nvidia-smi thread: {e}")
                if should_save_benchmark:
                    context.save_to_log_file(prompt)
                GLOBAL_CONTEXT = None

        return wrapper_PromptExecutor_execute
    execution.PromptExecutor.execute = factory_PromptExecutor_execute(execution.PromptExecutor.execute)

def initialize_benchmark_hooks():
    hook_PromptExecutor_execute()
    hook_CFGGuider_sample()
    hook_load_torch_file()
    hook_load_state_dict()
    hook_LoadedModel_model_load()
    hook_VAE()
    hook_CLIP()

def call_nvidia_smi(query_list: list[str], decode=True, set_nvidia_smi_error=False, print_error=False, raise_error=False) -> Union[str, None]:
    global NVIDIA_SMI_ERROR
    try:
        return subprocess.check_output(query_list, stderr=subprocess.STDOUT).decode("utf-8") if decode else subprocess.check_output(query_list)
    except subprocess.CalledProcessError as e:
        to_log = f"{e.returncode}:{e.output}"
        if set_nvidia_smi_error:
            NVIDIA_SMI_ERROR = to_log
        if print_error:
            logging.error(f"Error calling nvidia-smi: {to_log}")
        if raise_error:
            raise Exception(to_log)
        return None
    except Exception as e:
        to_log = f"{e}"
        if set_nvidia_smi_error:
            NVIDIA_SMI_ERROR = to_log
        if print_error:
            logging.error(f"Error calling nvidia-smi: {to_log}")
        if raise_error:
            raise Exception(to_log)
        return None

def call_psutil(include_prefix=True):
    memory_info = psutil.virtual_memory()
    prefix = "psutil:" if include_prefix else ""
    return f"{prefix}{time.perf_counter()},{memory_info.used},{memory_info.total}"

class BenchmarkExtension(ComfyExtension):
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        global INITIAL_NVIDIA_SMI_QUERY, ENABLE_NVIDIA_SMI_DATA, NVIDIA_SMI_ERROR, INFO_NVIDIA_SMI_QUERY, INITIAL_PSUTIL_QUERY
        initialize_benchmark_hooks()
        INITIAL_PSUTIL_QUERY = call_psutil(include_prefix=False)
        if comfy.model_management.is_nvidia():
            # get current memory usage
            try:
                INITIAL_NVIDIA_SMI_QUERY = call_nvidia_smi(_nvidia_smi_query_list, set_nvidia_smi_error=True, print_error=True)
                INFO_NVIDIA_SMI_QUERY = call_nvidia_smi(_info_nvidia_smi_query_list, set_nvidia_smi_error=True, print_error=True)
                if INITIAL_NVIDIA_SMI_QUERY is not None and INFO_NVIDIA_SMI_QUERY is not None:
                    ENABLE_NVIDIA_SMI_DATA = True
            except Exception as e:
                # NOTE: this should never happen, but just in case
                logging.error(f"Error getting initial nvidia smi query: {e}")
                NVIDIA_SMI_ERROR = f"{e}"
        return [BenchmarkWorkflow]

async def comfy_entrypoint():
    return BenchmarkExtension()