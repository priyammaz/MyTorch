import os
import yaml
import argparse
import sys
from pathlib import Path
import questionary
import shutil
from .banner import banner

custom_style_fancy = questionary.Style([
    ("highlighted", "fg:#00ff88 bold"), 
])

DEFAULT_CONFIG_PATH = Path(os.path.join(Path.home(), ".cache", "mytorch", "default_config.yaml"))

DEFAULT_CONFIG_ENV = {
    "distributed": False, 
    "num_gpus": 1, 
    "gpu_indices": "all",
    "master_addr": "127.0.0.1", 
    "master_port": "13333",
    "use_fused": "False",
    "triton_autotune": "none"
}

terminal_width = shutil.get_terminal_size().columns

def get_config_path():
    return DEFAULT_CONFIG_PATH

def save_config(config):
    """Save configuration to YAML file."""
    config_path = get_config_path()
    
    # Create directory if it doesn't exist
    config_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        print(f"\nConfiguration saved to {config_path}")
        return True
    except Exception as e:
        print(f"Error: Failed to save config to {config_path}: {e}")
        return False
    
def load_config(config_path=None):

    if config_path is None:
        config_path = get_config_path()
        using_default = True
    else:
        config_path = Path(config_path)
        using_default = False
    
    if not config_path.exists():
        if not using_default:
            # User specified a custom path that doesn't exist
            raise FileNotFoundError(f"Config file not found: {config_path}")
        # Default config doesn't exist, warn and return defaults
        print("WARNING: No default configuration found!")
        print("\nUsing built-in defaults. To create a config file, run:")
        print("  mytorchrun config")
        print("\nOr specify a custom config with:")
        print("  mytorchrun launch --config path/to/config.yaml\n")
        return DEFAULT_CONFIG_ENV.copy()
    
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config if config else DEFAULT_CONFIG_ENV.copy()
    except Exception as e:
        print(f"Warning: Failed to load config from {config_path}: {e}")
        return DEFAULT_CONFIG_ENV.copy()

def test():
    """Handle test subcommand."""
    from pathlib import Path
    
    # Get the tests directory
    mytorch_root = Path(__file__).parent
    tests_dir = mytorch_root / "tests"
    
    # Load config
    config = load_config()
    num_gpus = config.get("num_gpus", 1)
    compute_environment = config.get("compute_environment", "No Distributed Training")
    mixed_precision = config.get("mixed_precision", "No")
    
    print("=" * terminal_width)
    print("Running MyTorch Environment Tests")
    print("=" * terminal_width)
    print(f"\nConfiguration:")
    print(f"  Compute Environment: {compute_environment}")
    print(f"  Number of GPUs: {num_gpus}")
    print(f"  Mixed Precision: {mixed_precision}")
    print("=" * terminal_width)
    
    test_number = 1
    
    # Test 1: NCCL AllReduce (only if multi-GPU)
    if compute_environment == "Multi-GPU Training":
        allreduce_test_script = tests_dir / "all_reduce_test_w_launch.py"
        
        if not allreduce_test_script.exists():
            print(f"\nWarning: AllReduce test script not found at {allreduce_test_script}")
        else:
            print(f"\nTest {test_number}: NCCL AllReduce Communication Test")
            print("-" * terminal_width)
            print(f"Testing with {num_gpus} GPUs...")
            
            from mytorch.distributed.launch import main as launch_main
            
            sys.argv = [
                'mytorchrun',
                '--num_gpus', str(num_gpus),
                str(allreduce_test_script),
                '--master_addr', config.get("master_addr", "127.0.0.1"),
                '--master_port', config.get("master_port", "13333")
            ]
            
            try:
                launch_main()
                print(f"\n✓ Test {test_number} passed: NCCL AllReduce")
            except Exception as e:
                print(f"\n✗ Test {test_number} failed: {e}")
                sys.exit(1)
            
            test_number += 1
    
    # Test 2: Training Pipeline - No mixed precision, No gradient accumulation
    training_test_script = tests_dir / "test_training.py"
    
    if not training_test_script.exists():
        print(f"\nError: Training test script not found at {training_test_script}")
        sys.exit(1)
    
    print(f"\nTest {test_number}: Training Pipeline (No Mixed Precision, No Grad Accum)")
    print("-" * terminal_width)
    
    from mytorch.distributed.launch import main as launch_main
    
    sys.argv = [
        'mytorchrun',
        '--num_gpus', str(num_gpus),
        str(training_test_script),
        '--master_addr', config.get("master_addr", "127.0.0.1"),
        '--master_port', config.get("master_port", "13333")
    ]
    
    try:
        launch_main()
        print(f"\n✓ Test {test_number} passed: Basic training pipeline")
    except Exception as e:
        print(f"\n✗ Test {test_number} failed: {e}")
        sys.exit(1)
    
    test_number += 1
    
    # Test 3: Training Pipeline - With gradient accumulation
    print(f"\nTest {test_number}: Training Pipeline (Gradient Accumulation)")
    print("-" * terminal_width)
    
    sys.argv = [
        'mytorchrun',
        '--num_gpus', str(num_gpus),
        str(training_test_script),
        '--master_addr', config.get("master_addr", "127.0.0.1"),
        '--master_port', config.get("master_port", "13333"),
        '--gradient_accumulation_steps', '4'
    ]
    
    try:
        launch_main()
        print(f"\n✓ Test {test_number} passed: Gradient accumulation")
    except Exception as e:
        print(f"\n✗ Test {test_number} failed: {e}")
        sys.exit(1)
    
    test_number += 1
    
    # Test 4: Training Pipeline - With mixed precision (if configured)
    if mixed_precision == "fp16":
        print(f"\nTest {test_number}: Training Pipeline (Mixed Precision fp16)")
        print("-" * terminal_width)
        
        sys.argv = [
            'mytorchrun',
            '--num_gpus', str(num_gpus),
            str(training_test_script),
            '--master_addr', config.get("master_addr", "127.0.0.1"),
            '--master_port', config.get("master_port", "13333"),
            '--mixed_precision'
        ]
        
        try:
            launch_main()
            print(f"\n✓ Test {test_number} passed: Mixed precision training")
        except Exception as e:
            print(f"\n✗ Test {test_number} failed: {e}")
            sys.exit(1)
        
        test_number += 1
        
        # Test 5: Training Pipeline - Mixed precision + Gradient accumulation
        print(f"\nTest {test_number}: Training Pipeline (Mixed Precision + Grad Accum)")
        print("-" * terminal_width)
        
        sys.argv = [
            'mytorchrun',
            '--num_gpus', str(num_gpus),
            str(training_test_script),
            '--master_addr', config.get("master_addr", "127.0.0.1"),
            '--master_port', config.get("master_port", "13333"),
            '--mixed_precision',
            '--gradient_accumulation_steps', '4'
        ]
        
        try:
            launch_main()
            print(f"\n✓ Test {test_number} passed: Mixed precision + gradient accumulation")
        except Exception as e:
            print(f"\n✗ Test {test_number} failed: {e}")
            sys.exit(1)
    
    # Final summary
    print("\n" + "=" * terminal_width)
    print("✓ ALL TESTS PASSED")
    print("=" * terminal_width)
    print("\nYour MyTorch environment is properly configured!")
    print()

def env():
    try:
        config = load_config()
    except FileNotFoundError as e:
        print(f"Run `mytorchrun config` to generate config files for environment")
        sys.exit(1)

    print("\n" + "-" * terminal_width)
    print("Launch Configuration:")
    print("-" * terminal_width)
    for key, value in config.items():
        print(f"  {key}: {value}")
    print("-" * terminal_width)

def launch():
    from mytorch.distributed.launch import main as launch_main

    # Parse CLI args to see what was explicitly provided
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--config", type=str, default=None, help="Path to config file")
    parser.add_argument("--num_gpus", type=int, default=None)
    parser.add_argument("--master_addr", type=str, default=None)
    parser.add_argument("--master_port", type=str, default=None)
    
    # Parse only what we care about, keep the rest
    args, remaining = parser.parse_known_args(sys.argv[1:])
    
    # Load config (custom path or default)
    try:
        config = load_config(args.config)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)
    
    # Set Environment Variables 
    os.environ["TRITON_AUTOTUNE_MODE"] = str(config.get("triton_autotune", "none"))
    os.environ["USE_FUSED_OPS"] = str(config.get("use_fused", "False"))
    gpu_indices = str(config.get("gpu_indices"))
    if gpu_indices != "all":
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_indices

    print(banner)

    print("\n" + "-" * terminal_width)
    print("Launch Configuration:")
    print("-" * terminal_width)
    for key, value in config.items():
        print(f"  {key}: {value}")
    print("-" * terminal_width)

    # Build final args: CLI overrides config
    final_args = []

    # Add num_gpus (CLI overrides config)
    num_gpus = args.num_gpus if args.num_gpus is not None else config.get("num_gpus", 1)
    final_args.extend(["--num_gpus", str(num_gpus)])
    
    # Add master_addr (CLI overrides config)
    master_addr = args.master_addr if args.master_addr is not None else config.get("master_addr", "127.0.0.1")
    final_args.extend(["--master_addr", master_addr])
    
    # Add master_port (CLI overrides config)
    master_port = args.master_port if args.master_port is not None else config.get("master_port", "13333")
    final_args.extend(["--master_port", master_port])
    
    # Add all remaining args (including training script and its args)
    final_args.extend(remaining)
    
    # Set sys.argv for launch function
    sys.argv = ['mytorchrun'] + final_args
    launch_main()

def interactive_config():
    """Run interactive configuration setup."""
    print(banner)
    print("-" * terminal_width)

    config = {}

    machine_type = questionary.select(
        "What type of machine are you using?", 
        choices=[
            "No Distributed Training",
            "Multi-GPU Training"
        ],
        style=custom_style_fancy
    ).ask()

    config["compute_environment"] = machine_type

    mixed_precision = questionary.select(
        "Do you want to use mixed precision training?",
        choices=[
            "No", 
            "fp16"
        ],
        style=custom_style_fancy
    ).ask()
    
    fused_ops = questionary.select(
        "Do you want to use Fused Operations (Requires Triton Install)",
        choices=[
            "Yes",
            "No"
        ],
        style=custom_style_fancy
    ).ask()

    triton_autotune = "No"
    if fused_ops == "Yes":
        triton_autotune = questionary.select(
            "Do You Want to Use Triton Autotuning?",
            choices=[
                "Yes",
                "No"
            ],
            style=custom_style_fancy
    ).ask()

    config["mixed_precision"] = mixed_precision
    config["num_gpus"] = DEFAULT_CONFIG_ENV["num_gpus"]
    config["use_fused"] = "True" if fused_ops == "Yes" else "False"
    config["triton_autotune"] = "max" if triton_autotune == "Yes" else "none"

    if machine_type == "Multi-GPU Training":

        num_gpus = questionary.text(
            "How many GPU's woud you like to use?",
            validate=lambda text: True if text.isdigit() and int(text) > 1 else "Distributed training needs > 1 GPUs "
        ).ask()

        config["num_gpus"] = num_gpus

        gpu_indices = questionary.text(
            "What GPU(s) (by id) should be used for training as a comma-separated list? [all]"
            ).ask()
        
        if len(gpu_indices) == 0:
            gpu_indices = DEFAULT_CONFIG_ENV["gpu_indices"] 

        config["gpu_indices"] = gpu_indices     

        master_address = questionary.text(
            "Master Address",
            instruction="(Press Enter for default: 127.0.0.1)",
        ).ask()

        if len(master_address) == 0:
            master_address = DEFAULT_CONFIG_ENV["master_addr"]

        config["master_address"] = master_address
        
        master_port = questionary.text( 
            "Master Port",
            instruction="(Press Enter for default: 13333)",
        ).ask()

        if len(master_port) == 0:
            master_port = DEFAULT_CONFIG_ENV["master_port"]

        config["master_port"] = master_port

    print("\n" + "-" * terminal_width)
    print("Configuration Summary:")
    print("-" * terminal_width)
    for key, value in config.items():
        print(f"  {key}: {value}")
    print("-" * terminal_width)

    confirm = questionary.confirm(
        "Save this configuration?",
        default=True
    ).ask()

    if confirm is None or not confirm:
        print("\nConfiguration not saved.")
        return False
    
    if save_config(config):
        print("\nConfiguration complete!")
        print(f"  You can edit {get_config_path()} manually or run 'mytorchrun config' again.")
        return True

def main():

    parser = argparse.ArgumentParser(
        prog="mytorchrun"
    )

    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    subparsers.add_parser('launch', 
                          help='Launch distributed training',
                          add_help=False)
    subparsers.add_parser('test', 
                         help='Run environment tests')
    subparsers.add_parser('config', 
                         help='Run tests')
    subparsers.add_parser("env", 
                          help="See current env setup")
    
    args, remaining = parser.parse_known_args()

    if args.command == "launch":
        sys.argv = ['mytorchrun'] + remaining  # Fix: add dummy script name
        launch()

    elif args.command == "config":
        interactive_config()

    elif args.command == "test":
        test()
    
    elif args.command == "env":
        env()

    else:
        parser.print_help()
        sys.exit(1)