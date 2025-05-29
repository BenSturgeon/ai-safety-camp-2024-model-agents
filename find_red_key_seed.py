import sys
import os
import random

# Add src to path to allow importing utils
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

try:
    from utils import heist
    # Assuming ENTITY_TYPES and ENTITY_COLORS are defined in create_intervention_mazes or heist
    # Let's try importing from where they are most likely defined based on previous context
    from utils.create_intervention_mazes import ENTITY_TYPES, ENTITY_COLORS
except ImportError as e:
    print(f"Error importing necessary modules: {e}")
    print("Please ensure that utils/heist.py and utils/create_intervention_mazes.py exist and are accessible.")
    print("You might need to run this script from the root of the project directory.")
    sys.exit(1)

def find_seed_with_red_key(max_attempts=10000):
    """
    Attempts to find a start_level (seed) for Procgen Heist that generates
    an environment containing a red key.
    """
    print(f"Searching for a seed (start_level) that generates a red key (max {max_attempts} attempts)...")
    for attempt in range(max_attempts):
        start_level_seed = random.randint(1, 2**30) # Procgen seeds are integers
        
        venv = None
        try:
            # Create a heist environment with the specific seed
            # num_levels=0 makes start_level act as a specific seed
            venv = heist.create_venv(
                num=1, 
                start_level=start_level_seed, 
                num_levels=1,
                distribution_mode="easy" # memory mode is often used with fixed levels
            )
            state = heist.state_from_venv(venv, 0)

            if state.entity_exists(ENTITY_TYPES["key"], ENTITY_COLORS["red"]):
                print(f"\nSUCCESS: Found seed with red key!")
                print(f"  Seed (start_level): {start_level_seed}")
                print(f"  Attempt number: {attempt + 1}")
                return start_level_seed
            
            if (attempt + 1) % 100 == 0:
                print(f"  Attempt {attempt + 1}/{max_attempts}...")

        except Exception as e:
            print(f"  Warning: Encountered an error creating/checking environment for seed {start_level_seed}: {e}")
        finally:
            if venv:
                venv.close()
                
    print(f"\nFAILURE: Could not find a seed with a red key after {max_attempts} attempts.")
    return None

if __name__ == "__main__":
    found_seed = find_seed_with_red_key()
    if found_seed is not None:
        print(f"\nTo use this seed in your experiments, pass '--start_level {found_seed}' to the script that creates the environment.") 