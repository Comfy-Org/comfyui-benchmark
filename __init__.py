import time
import datetime
import logging
import os
import json
import argparse

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


VERSION = 0
# currently, for simplicity we use a global variable to store the execution context;
# this allows us to access the execution context from all hooks without having to remake the hooks at runtime
GLOBAL_CONTEXT = None
ENABLE_NVIDIA_SMI_DATA = False
INITIAL_NVIDIA_SMI_QUERY = None
INFO_NVIDIA_SMI_QUERY = None
NVIDIA_SMI_ERROR = None


nvidia_smi_query = ["timestamp", "memory.used", "memory.total", "utilization.gpu", "utilization.memory", "power.draw", "power.draw.instant", "power.limit", "pcie.link.gen.current", "pcie.link.gen.max", "pcie.link.width.current", "pcie.link.width.max"]
_nvidia_smi_query_list = ["nvidia-smi", "--query-gpu=" + ",".join(nvidia_smi_query), "--format=csv,noheader,nounits"]

info_nvidia_smi_query = ["name", "count", "driver_version", "display_attached", "display_active", "vbios_version", "power.management"]
_info_nvidia_smi_query_list = ["nvidia-smi", "--query-gpu=" + ",".join(info_nvidia_smi_query), "--format=csv,noheader,nounits"]

# For NVIDIA devices, during the benchmark setup a process to call nvidia-smi regularly (or with varying intervals)
def nvidia_smi_thread(out_queue: Queue, in_queue: Queue):
    logging.info("Starting nvidia-smi thread")
    while True:
        try:
            out_queue.put(subprocess.check_output(_nvidia_smi_query_list).decode("utf-8"))
        except Exception as e:
            logging.error(f"Breaking out of nvidia-smi thread due to {e}")
            break
        try:
            item = in_queue.get(timeout=0.25)
            if item == "stop":
                break
        except Empty:
            pass
        except Exception as e:
            logging.error(f"Breaking out of nvidia-smi thread reading in_queue due to {e}")
            break
    logging.info("Exiting nvidia-smi thread")

def create_nvidia_smi_thread():
    out_queue = Queue()
    in_queue = Queue()
    thread = Thread(target=nvidia_smi_thread, args=(out_queue, in_queue))
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

class ExecutionContext:
    def __init__(self, workflow_name: str=None):
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        if workflow_name is None:
            self.workflow_name = f"{timestamp}"
        else:
            self.workflow_name = f"{workflow_name}_{timestamp}"
        self.version = VERSION
        self.device_name = comfy.model_management.get_torch_device_name(comfy.model_management.get_torch_device())
        self.benchmark_data: dict[str] = {}
        self.load_torch_file_data: list[dict[str]] = []
        self.model_load_data: list[dict[str]] = []
        self.sampling_data: list[dict[str]] = []
        self.vae_data: list[dict[str,dict[str]]] = {
            "encode": [],
            "decode": []
        }
        self.startup_args = get_provided_args(comfy.cli_args.args, comfy.cli_args.parser)
        self.nvidia_smi_data_info: dict[str, str] = {}
        self.nvidia_smi_data: list[str] = []
        self._create_nvidia_smi_data_info()

    def _create_nvidia_smi_data_info(self):
        self.nvidia_smi_data_info["_info_nvidia_smi_query_params"] = ", ".join(info_nvidia_smi_query)
        self.nvidia_smi_data_info["_info_nvidia_smi_query"] = INFO_NVIDIA_SMI_QUERY
        self.nvidia_smi_data_info["_nvidia_smi_query_params"] = ", ".join(nvidia_smi_query)
        self.nvidia_smi_data_info["_initial_nvidia_smi_query"] = INITIAL_NVIDIA_SMI_QUERY
        self.nvidia_smi_data_info["_nvidia_smi_error"] = NVIDIA_SMI_ERROR

    def save_to_log_file(self, print_data: bool=False):
        output_dir = folder_paths.get_output_directory()
        benchmark_dir = os.path.join(output_dir, "benchmark")
        os.makedirs(benchmark_dir, exist_ok=True)
        benchmark_file = os.path.join(benchmark_dir, f"{self.workflow_name}.json")
        try:
            with open(benchmark_file, "w") as f:
                json.dump(self.__dict__, f, indent=4, ensure_ascii=False, default=json_func)
            if print_data:
                for key, value in self.benchmark_data.items():
                    if key.startswith("_"):
                        continue
                    logging.info(f"Benchmark: {key}: {value}")
            logging.info(f"Benchmark: {self.workflow_name} saved to {benchmark_file}")
        except Exception as e:
            logging.error(f"Error saving benchmark file {benchmark_file}: {e}")

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
                except Exception as _:
                    valid_timing = False
                    raise
                finally:
                    end_time = time.perf_counter()
                    context.vae_data["decode"].append({
                        "elapsed_time": end_time - start_time,
                        "start_time": start_time,
                        "valid_timing": valid_timing
                    })
                return func(*args, **kwargs)
            return wrapper_VAE_decode
        comfy.sd.VAE.decode = factory_VAE_decode(comfy.sd.VAE.decode)
    hook_VAE_encode()
    hook_VAE_decode()

def hook_LoadedModel_model_load():
    def factory_LoadedModel_model_load(func):
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
                context.model_load_data.append({
                    "model_name": str(args[0].model.model.__class__.__name__),
                    "elapsed_time": end_time - start_time,
                    "start_time": start_time,
                    "valid_timing": valid_timing
                })
        return wrapper_LoadedModel_model_load
    comfy.model_management.LoadedModel.model_load = factory_LoadedModel_model_load(comfy.model_management.LoadedModel.model_load)

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
                context.load_torch_file_data.append({
                    "ckpt": args[0],
                    "elapsed_time": end_time - start_time,
                    "start_time": start_time,
                    "valid_timing": valid_timing
                })
        return wrapper_load_torch_file
    comfy.utils.load_torch_file = factory_load_torch_file(comfy.utils.load_torch_file)

def hook_CFGGuider_sample():
    def add_sampling_wrappers(model_options: dict, context: ExecutionContext, temp_dict: dict[str]):
        def factory_predict_noise(c, temp_dict: dict[str]):
            def wrapper_predict_noise(executor, *args, **kwargs):
                temp_dict.setdefault("_iteration_times", [])
                try:
                    start_time = time.perf_counter()
                    return executor(*args, **kwargs)
                finally:
                    end_time = time.perf_counter()
                    temp_dict["_iteration_times"].append(end_time - start_time)
            return wrapper_predict_noise
        comfy.patcher_extension.add_wrapper_with_key(comfy.patcher_extension.WrappersMP.PREDICT_NOISE, "benchmark_sampling", factory_predict_noise(context, temp_dict),
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
                add_sampling_wrappers(model_options, GLOBAL_CONTEXT, temp_dict)
                guider.model_options = model_options
                cfg_guider_start_time = time.perf_counter()
                return func(*args, **kwargs)
            finally:
                cfg_guider_end_time = time.perf_counter()
                temp_dict["cfg_guider_elapsed_time"] = cfg_guider_end_time - cfg_guider_start_time
                temp_dict["_cfg_guider_start_time"] = cfg_guider_start_time
                temp_dict["_cfg_guider_end_time"] = cfg_guider_end_time
                if "_iteration_times" in temp_dict:
                    temp_dict["average_iteration_time"] = sum(temp_dict["_iteration_times"]) / len(temp_dict["_iteration_times"])
                else:
                    temp_dict["average_iteration_time"] = -1
                GLOBAL_CONTEXT.sampling_data.append(temp_dict)
                guider.model_options = orig_model_options
        return wrapper_CFGGuider_sample
    comfy.samplers.CFGGuider.sample = factory_CFGGuider_sample(comfy.samplers.CFGGuider.sample)

def hook_PromptExecutor_execute():
    def factory_PromptExecutor_execute(func):
        '''
        Create wrapper function that will time the total execution time for a workflow.
        '''
        def wrapper_PromptExecutor_execute(*args, **kwargs):
            global GLOBAL_CONTEXT, ENABLE_NVIDIA_SMI_DATA, INITIAL_NVIDIA_SMI_QUERY, INFO_NVIDIA_SMI_QUERY, NVIDIA_SMI_ERROR
            args = args
            kwargs = kwargs
            # create execution context
            context = ExecutionContext(workflow_name="benchmark")
            GLOBAL_CONTEXT = context
            # if its an nvidia card, we can do overall memory usage tracking via nvidia-smi calls
            thread_started = False
            if ENABLE_NVIDIA_SMI_DATA:
                out_queue, in_queue, thread = create_nvidia_smi_thread()
                thread_started = True
            start_datetime = datetime.datetime.now().strftime("%Y/%m/%d %H:%M:%S.%f")
            try:
                start_time = time.perf_counter()
                result = func(*args, **kwargs)
                end_time = time.perf_counter()
                context.benchmark_data["execution_elapsed_time"] = end_time - start_time
                context.benchmark_data["_workflow_start_datetime"] = start_datetime
                context.benchmark_data["_workflow_start_time"] = start_time
                context.benchmark_data["_workflow_end_time"] = end_time
                return result
            finally:
                if thread_started:
                    try:
                        in_queue.put("stop")
                        thread.join()
                        while not out_queue.empty():
                            context.nvidia_smi_data.append(out_queue.get_nowait())
                    except Exception as e:
                        logging.error(f"Error stopping nvidia-smi thread: {e}")
                context.save_to_log_file()
                GLOBAL_CONTEXT = None

        return wrapper_PromptExecutor_execute
    execution.PromptExecutor.execute = factory_PromptExecutor_execute(execution.PromptExecutor.execute)

def initialize_benchmark_hooks():
    hook_PromptExecutor_execute()
    hook_CFGGuider_sample()
    hook_load_torch_file()
    hook_LoadedModel_model_load()
    hook_VAE()


class BenchmarkExtension(ComfyExtension):
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        global INITIAL_NVIDIA_SMI_QUERY, ENABLE_NVIDIA_SMI_DATA, NVIDIA_SMI_ERROR, INFO_NVIDIA_SMI_QUERY
        initialize_benchmark_hooks()
        if comfy.model_management.is_nvidia():
            # get current memory usage
            try:
                INITIAL_NVIDIA_SMI_QUERY = subprocess.check_output(_nvidia_smi_query_list).decode("utf-8")
                INFO_NVIDIA_SMI_QUERY = subprocess.check_output(_info_nvidia_smi_query_list).decode("utf-8")
                ENABLE_NVIDIA_SMI_DATA = True
            except Exception as e:
                logging.error(f"Error getting initial nvidia smi query: {e}")
                NVIDIA_SMI_ERROR = f"{e}"
        return []

async def comfy_entrypoint():
    return BenchmarkExtension()
