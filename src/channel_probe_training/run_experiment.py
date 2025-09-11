#!/usr/bin/env python3
"""
Main experiment runner for channel-wise probe training.
Orchestrates data collection and probe training pipeline.
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import argparse
import logging
from datetime import datetime
import json
import subprocess
import time


def setup_logging(experiment_name="channel_probe_experiment"):
    """Setup logging for the experiment."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join(os.path.dirname(__file__), 'logs')
    os.makedirs(log_dir, exist_ok=True)
    
    log_file = os.path.join(log_dir, f'{experiment_name}_{timestamp}.log')
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f"Logging to {log_file}")
    return logger, timestamp


def run_data_collection(num_samples, entity_types, logger):
    """Run the data collection script."""
    logger.info("="*60)
    logger.info("PHASE 1: Data Collection")
    logger.info("="*60)
    
    script_path = os.path.join(os.path.dirname(__file__), 'collect_empty_maze_data.py')
    
    # Modify the script to accept command line arguments
    cmd = [
        sys.executable, script_path,
        '--num_samples', str(num_samples),
        '--entity_types', entity_types
    ]
    
    logger.info(f"Running: {' '.join(cmd)}")
    
    start_time = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.time() - start_time
    
    if result.returncode != 0:
        logger.error(f"Data collection failed:\n{result.stderr}")
        return False
    
    logger.info(f"Data collection completed in {elapsed:.1f} seconds")
    return True


def run_probe_training(max_channels, n_jobs, logger):
    """Run the probe training script."""
    logger.info("="*60)
    logger.info("PHASE 2: Probe Training")
    logger.info("="*60)
    
    script_path = os.path.join(os.path.dirname(__file__), 'train_channel_probes.py')
    
    cmd = [sys.executable, script_path]
    
    if max_channels:
        # Modify script to accept max_channels argument
        cmd.extend(['--max_channels', str(max_channels)])
    
    if n_jobs:
        cmd.extend(['--n_jobs', str(n_jobs)])
    
    logger.info(f"Running: {' '.join(cmd)}")
    
    start_time = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.time() - start_time
    
    if result.returncode != 0:
        logger.error(f"Probe training failed:\n{result.stderr}")
        return False
    
    logger.info(f"Probe training completed in {elapsed:.1f} seconds")
    return True


def update_scripts_for_cli():
    """Update the data collection and training scripts to accept CLI arguments."""
    
    # Update collect_empty_maze_data.py
    collect_script = os.path.join(os.path.dirname(__file__), 'collect_empty_maze_data.py')
    with open(collect_script, 'r') as f:
        content = f.read()
    
    if "argparse" not in content:
        # Add argparse to main function
        new_main = '''def main():
    # Setup argument parser
    parser = argparse.ArgumentParser(description='Collect empty maze dataset')
    parser.add_argument('--num_samples', type=int, default=5000,
                       help='Number of samples to collect')
    parser.add_argument('--entity_types', type=str, default='keys_and_gems',
                       choices=['keys_only', 'gems_only', 'keys_and_gems'],
                       help='Types of entities to include')
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    logger.info("="*60)
    logger.info("Collecting Empty Maze Dataset")
    logger.info("="*60)
    
    # Set random seeds
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Collect dataset
    observations, labels, label_map, metadata = collect_dataset(
        num_samples=args.num_samples,
        entity_types=args.entity_types,
        logger=logger
    )
    
    # Save dataset
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dataset_path = os.path.join(
        os.path.dirname(__file__), 
        f'empty_maze_dataset_{timestamp}.pkl'
    )
    
    with open(dataset_path, 'wb') as f:
        pickle.dump({
            'observations': observations,
            'labels': labels,
            'label_map': label_map,
            'metadata': metadata,
            'entity_types': args.entity_types,
            'num_samples': args.num_samples
        }, f)
    
    logger.info(f"\\nDataset saved to: {dataset_path}")
    logger.info(f"Total samples: {len(observations)}")
    logger.info("Data collection complete!")'''
        
        # Add import
        content = content.replace("import sys", "import sys\nimport argparse")
        # Replace main function
        content = content.replace("def main():", new_main, 1)
        content = content.replace("num_samples = 5000  # Start with smaller dataset for testing", "")
        content = content.replace("num_samples=num_samples,", "num_samples=args.num_samples,")
        content = content.replace("entity_types='keys_and_gems',", "entity_types=args.entity_types,")
        
        with open(collect_script, 'w') as f:
            f.write(content)
    
    # Update train_channel_probes.py
    train_script = os.path.join(os.path.dirname(__file__), 'train_channel_probes.py')
    with open(train_script, 'r') as f:
        content = f.read()
    
    if "argparse" not in content:
        # Add argparse
        content = content.replace("import sys", "import sys\nimport argparse")
        
        # Update main function
        old_main_start = "def main():"
        new_main_start = '''def main():
    # Setup argument parser
    parser = argparse.ArgumentParser(description='Train channel-wise probes')
    parser.add_argument('--max_channels', type=int, default=None,
                       help='Maximum number of channels to train (for testing)')
    parser.add_argument('--n_jobs', type=int, default=4,
                       help='Number of parallel jobs')
    args = parser.parse_args()
    '''
        
        content = content.replace(old_main_start, new_main_start, 1)
        content = content.replace("max_channels = None  # Set to small number for testing, None for full training",
                                 "max_channels = args.max_channels")
        content = content.replace("n_jobs=4,", "n_jobs=args.n_jobs,")
        
        with open(train_script, 'w') as f:
            f.write(content)


def main():
    parser = argparse.ArgumentParser(description='Run channel-wise probe training experiment')
    parser.add_argument('--num_samples', type=int, default=1000,
                       help='Number of samples to collect')
    parser.add_argument('--entity_types', type=str, default='keys_and_gems',
                       choices=['keys_only', 'gems_only', 'keys_and_gems'],
                       help='Types of entities to include in empty mazes')
    parser.add_argument('--max_channels', type=int, default=None,
                       help='Maximum number of channels to train (None for all)')
    parser.add_argument('--n_jobs', type=int, default=4,
                       help='Number of parallel jobs for training')
    parser.add_argument('--skip_collection', action='store_true',
                       help='Skip data collection and use existing dataset')
    parser.add_argument('--test_mode', action='store_true',
                       help='Run in test mode with small samples')
    
    args = parser.parse_args()
    
    # Setup logging
    logger, timestamp = setup_logging()
    
    logger.info("="*60)
    logger.info("Channel-wise Probe Training Experiment")
    logger.info("="*60)
    logger.info(f"Configuration:")
    logger.info(f"  Num samples: {args.num_samples}")
    logger.info(f"  Entity types: {args.entity_types}")
    logger.info(f"  Max channels: {args.max_channels}")
    logger.info(f"  Parallel jobs: {args.n_jobs}")
    logger.info(f"  Test mode: {args.test_mode}")
    
    # Test mode adjustments
    if args.test_mode:
        args.num_samples = min(args.num_samples, 100)
        args.max_channels = min(args.max_channels or 10, 10)
        logger.info("Test mode: Reduced samples and channels")
    
    # Update scripts to accept CLI arguments
    logger.info("Updating scripts for CLI support...")
    update_scripts_for_cli()
    
    # Phase 1: Data Collection
    if not args.skip_collection:
        success = run_data_collection(args.num_samples, args.entity_types, logger)
        if not success:
            logger.error("Data collection failed. Exiting.")
            return 1
    else:
        logger.info("Skipping data collection, using existing dataset")
    
    # Phase 2: Probe Training
    success = run_probe_training(args.max_channels, args.n_jobs, logger)
    if not success:
        logger.error("Probe training failed. Exiting.")
        return 1
    
    # Save experiment configuration
    config = {
        'timestamp': timestamp,
        'num_samples': args.num_samples,
        'entity_types': args.entity_types,
        'max_channels': args.max_channels,
        'n_jobs': args.n_jobs,
        'test_mode': args.test_mode
    }
    
    config_file = os.path.join(os.path.dirname(__file__), f'experiment_config_{timestamp}.json')
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2)
    
    logger.info(f"Experiment configuration saved to {config_file}")
    logger.info("="*60)
    logger.info("Experiment completed successfully!")
    logger.info("="*60)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())