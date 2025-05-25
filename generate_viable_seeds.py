import sys
import os
import random

# Add src to path to allow importing utils
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

try:
    from utils import heist
    from utils.create_intervention_mazes import ENTITY_TYPES, ENTITY_COLORS
except ImportError as e:
    print(f"Error importing necessary modules: {e}")
    print("Please ensure that utils/heist.py and utils/create_intervention_mazes.py exist and are accessible.")
    print("You might need to run this script from the root of the project directory.")
    sys.exit(1)

def generate_seeds_with_all_keys(num_seeds_to_find=500, max_attempts_total=1000000, output_file="viable_seeds.txt"):
    """
    Attempts to find a specified number of start_levels (seeds) for Procgen Heist
    that generate environments containing red, green, AND blue keys.
    Saves the found seeds to a file.
    """
    print(f"Searching for {num_seeds_to_find} seeds (start_level) that generate red, green, and blue keys.")
    print(f"Max total attempts: {max_attempts_total}")
    
    found_seeds = []
    attempts_done = 0

    while len(found_seeds) < num_seeds_to_find and attempts_done < max_attempts_total:
        start_level_seed = random.randint(1, 2**30) # Procgen seeds are integers
        attempts_done += 1
        venv = None
        try:
            # Create a heist environment with the specific seed
            # num_levels=0 makes start_level act as a specific seed
            venv = heist.create_venv(
                num=1, 
                start_level=start_level_seed, 
                num_levels=1, # Ensures start_level acts as a unique seed
                distribution_mode="easy" # Consistent with how levels are typically fixed
            )
            state = heist.state_from_venv(venv, 0)

            has_red_key = state.entity_exists(ENTITY_TYPES["key"], ENTITY_COLORS["red"])
            has_green_key = state.entity_exists(ENTITY_TYPES["key"], ENTITY_COLORS["green"])
            has_blue_key = state.entity_exists(ENTITY_TYPES["key"], ENTITY_COLORS["blue"])

            if has_red_key and has_green_key and has_blue_key:
                if start_level_seed not in found_seeds: # Ensure uniqueness
                    found_seeds.append(start_level_seed)
                    print(f"  Found seed {len(found_seeds)}/{num_seeds_to_find}: {start_level_seed} (Attempt {attempts_done})")
            
            if attempts_done % 1000 == 0:
                print(f"  Attempt {attempts_done}/{max_attempts_total}... Found {len(found_seeds)} seeds so far.")

        except Exception as e:
            # print(f"  Warning: Encountered an error creating/checking environment for seed {start_level_seed}: {e}")
            # Suppressing individual errors to avoid cluttering output during long runs
            pass
        finally:
            if venv:
                venv.close()
    
    if len(found_seeds) >= num_seeds_to_find:
        print(f"\nSUCCESS: Found {len(found_seeds)} seeds with all three key types.")
    else:
        print(f"\nWARNING: Found {len(found_seeds)} seeds after {max_attempts_total} attempts (target was {num_seeds_to_find}).")

    if found_seeds:
        try:
            with open(output_file, 'w') as f:
                for seed in found_seeds:
                    f.write(str(seed) + '\n')
            print(f"Saved {len(found_seeds)} seeds to {output_file}")
        except IOError as e:
            print(f"Error saving seeds to file: {e}")
            print("Found seeds are:", found_seeds)
    
    return found_seeds

if __name__ == "__main__":
    print("Starting seed generation process...")
    viable_seeds_list = generate_seeds_with_all_keys(num_seeds_to_find=500)
    if viable_seeds_list:
        print(f"\nProcess complete. {len(viable_seeds_list)} seeds found and saved.")
    else:
        print("\nProcess complete. No viable seeds found with the current criteria/attempts.") 