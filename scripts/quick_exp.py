import os
import re
import time
import signal
import argparse
import subprocess
import json


"---------------------------------- Parser ----------------------------------"
parser = argparse.ArgumentParser(description="Quick experiment script.")
mode_helper = \
'''\
The mode of the script. Default is "func". 
Options are:
    "func": Benchmark for dReLU and activation functions.
    "snni": Benchmark for SNNI.
'''
parser.add_argument("-m", "--mode", type=str, default="func", help="The mode of the script. Default is \"func\".")
protocols_helper = \
'''\
The protocols to use. Default is "mizar".
Options are:
    "falcon": P-Falcon protocol, where sigmoid is replaced by spline version.
    "aegis": Aegis protocol, which is our baseline.
    "aegisopt": Aegis optimization protocol.
    "mizar": Mizar protocol, which is our new protocol.
'''
parser.add_argument("-p", "--protocols", type=str, nargs="+", default=["mizar"], help="The protocols to test. Default is [\"func\"].")
parser.add_argument("-s", "--enable_simulated_setup", action=argparse.BooleanOptionalAction)
parser.add_argument("-c", "--cuda_visible_devices", type=int, nargs="+", default=[0, 1, 2], help="Which GPU will be used for each party.")
parser.add_argument("-d", "--debug_mode", action=argparse.BooleanOptionalAction)
# Parameters for unittest.
# Parameters for snni.
parser.add_argument("--models", type=str, nargs="+", default=["lenet"], help="The model to use. Default is \"lenet\".")
parser.add_argument("--num_epochs", type=int, default=1, help="The number of epochs to run.")
parser.add_argument("--num_iterations", type=int, default=1, help="The number of iterations to run.")
parser.add_argument("--batch_size", type=int, default=1, help="The batch size to use.")


"---------------------------------- Config Manager ----------------------------------"
class ConfigManager:
    def __init__(self, config_path: str):
        self.load_config(config_path)
    
    def load_config(self, config_path):
        with open(config_path, "r") as f:
            self.config = json.load(f)
    
    def dump_config(self, dump_path: str):
        with open(dump_path, "w") as f:
            json.dump(self.config, f, indent=4)
    
    def update_config(self, key: str, value):
        self.config[key] = value


"---------------------------------- Main ----------------------------------"
PROTOCOL = "MIZAR"

protocols_map = {
    "falcon":   "piranha-rss",
    "aegis":    "piranha-aegis",
    "aegisopt": "piranha-aegisopt",
    "mizar":    "piranha-mizar"
}

inference_network_map = {
    "falcon":    {
        "lenet": "files/models/lenet-norelu-default.json",
        "vgg16": "files/models/vgg16-cifar10-norelu-default.json"
    },
    "aegis":    {
        "lenet": "files/models/lenet-norelu-aegis.json",
        "vgg16": "files/models/vgg16-cifar10-norelu-aegis.json"
    },
    "aegisopt":    {
        "lenet": "files/models/lenet-norelu-aegis.json",
        "vgg16": "files/models/vgg16-cifar10-norelu-aegis.json"
    },
    "mizar":    {
        "lenet": "files/models/lenet-norelu-default.json",
        "vgg16": "files/models/vgg16-cifar10-norelu-default.json"
    }
}

train_network_map = {
    "lenet":    "files/models/lenet-norelu-aegis.json",
    "vgg16":    "files/models/vgg16-cifar10-norelu-aegis.json"
}

def quick_func_exp(protocols: list[str], cuda_visible_devices: list[int], use_simulated_setup: bool = False, debug_mode: bool = False):
    '''
    Run the quick func benchmark experiment.
    
    Parameters:
        protocols: The protocols to test. For {protocol}, we will find the binary file in program root dir.
        cuda_visible_devices: The CUDA visible devices to use. For {cuda_visible_device}, we will set the CUDA_VISIBLE_DEVICES environment variable.
        use_simulated_setup: Whether to use the simulated setup. If True, we will use the offline binary file.
    '''
    if use_simulated_setup is None:
        use_simulated_setup = False
        
    for protocol in protocols:            
        timestamp = time.strftime("%Y%m%d-%H%M")
        
        # console infomation.
        print(f"-----------------------------------")
        print(f"Protocol:           {protocol}")
        print(f"Timestamp:          {timestamp}")
        print(f"Enable offline:     {use_simulated_setup}")
        print(f"Debug mode:         {debug_mode}")
        
        # create log directory if it doesn't exist
        log_dir = f"./output/func/{timestamp}-{protocol}/"
        if not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)
        
        # generate config file.
        config = ConfigManager("./template.json")
        config.update_config("run_unit_tests", True)
        config.update_config("unit_test_only", True)
        config.update_config("protocol", protocol)
        config.update_config("use_simulated_setup", use_simulated_setup)
        dump_dir = f"{log_dir}/config.json"
        config.dump_config(dump_dir)

        # Check if the binary file exists
        binary_file = f"./output/bin/{protocols_map[protocol]}" if not use_simulated_setup else f"./output/bin/{protocols_map[protocol]}-offline"
        if not os.path.exists(binary_file):
            raise Exception(f"Binary file {binary_file} not found.")
        
        with open(f"{log_dir}/p0.log", "w") as p0_log:
            p0 = subprocess.Popen([
                binary_file, 
                "-p", "0",
                "-c", dump_dir], 
                env={"CUDA_VISIBLE_DEVICES": f"{cuda_visible_devices[0]}"},
                stdout=p0_log)
        with open(f"{log_dir}/p1.log", "w") as p1_log:
            p1 = subprocess.Popen([
                binary_file, 
                "-p", "1",
                "-c", dump_dir], 
                env={"CUDA_VISIBLE_DEVICES": f"{cuda_visible_devices[1]}"},
                stdout=p1_log)
        with open(f"{log_dir}/p2.log", "w") as p2_log:
            p2 = subprocess.Popen([
                binary_file, 
                "-p", "2",
                "-c", dump_dir], 
                env={"CUDA_VISIBLE_DEVICES": f"{cuda_visible_devices[2]}"},
                stdout=p2_log)
        print(f"PID:                    {p0.pid} {p1.pid} {p2.pid}")
        
        # wait util the process end.
        start_time = time.time()
        while True:
            time.sleep(1)
            print(f"Time:               {time.time() - start_time:<4.2f}s", end="\r")
            if p0.poll() is not None and p1.poll() is not None and p2.poll() is not None:
                print("\nDone.")
                break

def quick_snni_exp(
    protocols: list[str], cuda_visible_devices: list[int], models: list[str], 
    num_epochs: int, num_iterations: int, batch_size: int, 
    use_simulated_setup: bool = False, debug_mode: bool = False):
    '''
    Run the quick snni benchmark experiment.
    
    Parameters:
        protocols: The protocols to test. For {params}, we will find the binary file in program root dir.
        cuda_visible_devices: The CUDA visible devices to use. For {cuda_visible_device}, we will set the CUDA_VISIBLE_DEVICES environment variable.
        models: The models to test. 
        num_epochs: The number of epochs to run.
        num_iterations: The number of iterations to run.
        batch_size: The batch size to use.
        use_simulated_setup: Whether to use the simulated setup. If True, we will use the offline binary file.
    '''
    if use_simulated_setup is None:
        use_simulated_setup = False
        
    parameters = [(protocol, model) for protocol in protocols for model in models]
    for protocol, model in parameters:            
        timestamp = time.strftime("%Y%m%d-%H%M")
        network_dir = inference_network_map[protocol][model]
        
        # console infomation.
        print(f"-----------------------------------")
        print(f"Protocol:               {protocol}")
        print(f"Timestamp:              {timestamp}")
        print(f"Model:                  {model}")
        print(f"Num Epochs:             {num_epochs}")
        print(f"Num Iterations:         {num_iterations}")
        print(f"Batch Size:             {batch_size}")
        print(f"Enable offline:         {use_simulated_setup}")
        print(f"Debug mode:             {debug_mode}")
        
        # create log directory if it doesn't exist
        log_dir = f"./output/snni/{timestamp}-{protocol}-{model}/"
        if not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)
        
        # generate config file.
        config = ConfigManager("./template.json")
        config.update_config("run_unit_tests", False)
        config.update_config("network", network_dir)
        config.update_config("custom_epochs", True)
        config.update_config("custom_epoch_count", num_epochs)
        config.update_config("custom_iterations", True)
        config.update_config("custom_iteration_count", num_iterations)
        config.update_config("custom_batch_size", True)
        config.update_config("custom_batch_size_count", batch_size)
        config.update_config("test_only", False)
        config.update_config("no_test", True)
        config.update_config("inference_only", True)
        config.update_config("eval_accuracy", False)
        config.update_config("protocol", protocol)
        config.update_config("use_simulated_setup", use_simulated_setup)
        config.update_config("eval_epoch_stats", True)
        config.update_config("eval_fw_peak_memory", True)
        config.update_config("eval_inference_stats", False)
        config.update_config("eval_train_stats", False)
        config.update_config("debug_print", False)
        if debug_mode:
            config.update_config("eval_inference_stats", True)
            config.update_config("eval_train_stats", True)
            config.update_config("debug_print", True)
            config.update_config("debug_all_forward", True)
            config.update_config("debug_all_backward", True)
        dump_dir = f"{log_dir}/config.json"
        config.dump_config(dump_dir)

        # Check if the binary file exists
        binary_file = f"./output/bin/{protocols_map[protocol]}" if not use_simulated_setup else f"./output/bin/{protocols_map[protocol]}-offline"
        if not os.path.exists(binary_file):
            raise Exception(f"Binary file {binary_file} not found.")
        
        with open(f"{log_dir}/p0.log", "w") as p0_log:
            p0 = subprocess.Popen([
                binary_file, 
                "-p", "0",
                "-c", dump_dir], 
                env={"CUDA_VISIBLE_DEVICES": f"{cuda_visible_devices[0]}"},
                stdout=p0_log)
        with open(f"{log_dir}/p1.log", "w") as p1_log:
            p1 = subprocess.Popen([
                binary_file, 
                "-p", "1",
                "-c", dump_dir], 
                env={"CUDA_VISIBLE_DEVICES": f"{cuda_visible_devices[1]}"},
                stdout=p1_log)
        with open(f"{log_dir}/p2.log", "w") as p2_log:
            p2 = subprocess.Popen([
                binary_file, 
                "-p", "2",
                "-c", dump_dir], 
                env={"CUDA_VISIBLE_DEVICES": f"{cuda_visible_devices[2]}"},
                stdout=p2_log)
        print(f"PID:                    {p0.pid} {p1.pid} {p2.pid}")
        
        # wait util the process end.
        start_time = time.time()
        while True:
            time.sleep(1)
            print(f"Time:               {time.time() - start_time:<4.2f}s", end="\r")
            if p0.poll() is not None and p1.poll() is not None and p2.poll() is not None:
                print("\nDone.")
                break
            
def quick_train_exp(
    protocols: list[str], cuda_visible_devices: list[int], models: list[str], 
    num_epochs: int, num_iterations: int, batch_size: int, 
    use_simulated_setup: bool = False, debug_mode: bool = False):
    '''
    Run the quick snni benchmark experiment.
    
    Parameters:
        protocols: The protocols to test. For {params}, we will find the binary file in program root dir.
        cuda_visible_devices: The CUDA visible devices to use. For {cuda_visible_device}, we will set the CUDA_VISIBLE_DEVICES environment variable.
        models: The models to test. 
        num_epochs: The number of epochs to run.
        num_iterations: The number of iterations to run.
        batch_size: The batch size to use.
        use_simulated_setup: Whether to use the simulated setup. If True, we will use the offline binary file.
    '''
    if use_simulated_setup is None:
        use_simulated_setup = False
        
    parameters = [(protocol, model) for protocol in protocols for model in models]
    for protocol, model in parameters:            
        timestamp = time.strftime("%Y%m%d-%H%M")
        network_dir = train_network_map[model]
        
        # console infomation.
        print(f"-----------------------------------")
        print(f"Protocol:               {protocol}")
        print(f"Timestamp:              {timestamp}")
        print(f"Model:                  {model}")
        print(f"Num Epochs:             {num_epochs}")
        print(f"Num Iterations:         {num_iterations}")
        print(f"Batch Size:             {batch_size}")
        print(f"Enable offline:         {use_simulated_setup}")
        print(f"Debug mode:             {debug_mode}")
        
        # create log directory if it doesn't exist
        log_dir = f"./output/train/{timestamp}-{protocol}-{model}/"
        if not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)
        
        # generate config file.
        config = ConfigManager("./template.json")
        config.update_config("run_unit_tests", False)
        config.update_config("network", network_dir)
        config.update_config("custom_epochs", True)
        config.update_config("custom_epoch_count", num_epochs)
        config.update_config("custom_iterations", True)
        config.update_config("custom_iteration_count", num_iterations)
        config.update_config("custom_batch_size", True)
        config.update_config("custom_batch_size_count", batch_size)
        config.update_config("test_only", False)
        config.update_config("no_test", True)
        config.update_config("inference_only", False)
        config.update_config("eval_accuracy", False)
        config.update_config("protocol", protocol)
        config.update_config("use_simulated_setup", use_simulated_setup)
        config.update_config("eval_epoch_stats", True)
        config.update_config("eval_epoch_stats", True)
        config.update_config("eval_fw_peak_memory", True)
        config.update_config("eval_inference_stats", False)
        config.update_config("eval_train_stats", False)
        config.update_config("debug_print", False)
        if debug_mode:
            config.update_config("eval_inference_stats", True)
            config.update_config("eval_train_stats", True)
            config.update_config("debug_print", True)
            config.update_config("debug_all_forward", True)
            config.update_config("debug_all_backward", True)
        dump_dir = f"{log_dir}/config.json"
        config.dump_config(dump_dir)

        # Check if the binary file exists
        binary_file = f"./output/bin/{protocols_map[protocol]}" if not use_simulated_setup else f"./output/bin/{protocols_map[protocol]}-offline"
        if not os.path.exists(binary_file):
            raise Exception(f"Binary file {binary_file} not found.")
        
        with open(f"{log_dir}/p0.log", "w") as p0_log:
            p0 = subprocess.Popen([
                binary_file, 
                "-p", "0",
                "-c", dump_dir], 
                env={"CUDA_VISIBLE_DEVICES": f"{cuda_visible_devices[0]}"},
                stdout=p0_log)
        with open(f"{log_dir}/p1.log", "w") as p1_log:
            p1 = subprocess.Popen([
                binary_file, 
                "-p", "1",
                "-c", dump_dir], 
                env={"CUDA_VISIBLE_DEVICES": f"{cuda_visible_devices[1]}"},
                stdout=p1_log)
        with open(f"{log_dir}/p2.log", "w") as p2_log:
            p2 = subprocess.Popen([
                binary_file, 
                "-p", "2",
                "-c", dump_dir], 
                env={"CUDA_VISIBLE_DEVICES": f"{cuda_visible_devices[2]}"},
                stdout=p2_log)
        print(f"PID:                    {p0.pid} {p1.pid} {p2.pid}")
        
        # wait util the process end.
        start_time = time.time()
        while True:
            time.sleep(1)
            print(f"Time:               {time.time() - start_time:<4.2f}s", end="\r")
            if p0.poll() is not None and p1.poll() is not None and p2.poll() is not None:
                print("\nDone.")
                break
                

if __name__ == "__main__":
    args = parser.parse_args()
    if args.mode == "func":
        quick_func_exp(args.protocols, args.cuda_visible_devices, args.enable_simulated_setup, args.debug_mode)
    elif args.mode == "snni":
        quick_snni_exp(args.protocols, args.cuda_visible_devices, args.models, 
                       args.num_epochs, args.num_iterations, args.batch_size, 
                       args.enable_simulated_setup, args.debug_mode)
    elif args.mode == "train":
        quick_train_exp(args.protocols, args.cuda_visible_devices, args.models, 
                       args.num_epochs, args.num_iterations, args.batch_size, 
                       args.enable_simulated_setup, args.debug_mode)
    else:
        raise ValueError(f"Unknown mode: {args.mode}. Please use \"func\" or \"snni\".")
    