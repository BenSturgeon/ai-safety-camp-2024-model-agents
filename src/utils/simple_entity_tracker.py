#!/usr/bin/env python3
"""
Simple entity collection tracker for Heist environment.
Just call update() each step to track what was collected.
"""

from typing import Dict, List, Tuple
from utils.heist import EnvState


class EntityTracker:
    """
    Simple tracker for entity collections in Heist environment.
    
    Usage:
        tracker = EntityTracker(initial_state)
        
        # Each step:
        collected = tracker.update(current_state, step_number)
        if collected:
            print(f"Collected: {collected}")
            
        # Get summary anytime:
        print(tracker.get_collection_order())  # [(5, 'blue_key'), (10, 'gem')]
    """
    
    ENTITY_NAMES = {
        3: "gem",
        4: "blue_key", 
        5: "green_key",
        6: "red_key"
    }
    
    def __init__(self, initial_state: EnvState):
        """Initialize tracker with the initial environment state."""
        self.previous_counts = self._get_counts(initial_state)
        self.collection_order = []
        
    def update(self, current_state: EnvState, step_number: int, done: bool = False) -> List[str]:
        """
        Update tracker with current state. Returns list of entities collected this step.
        
        Args:
            current_state: Current EnvState
            step_number: Current step number
            done: Whether the episode ended (important for gem detection)
            
        Returns:
            List of entity names collected this step (empty if none)
        """
        current_counts = self._get_counts(current_state)
        collected = []
        
        # Check each entity type
        for code, name in self.ENTITY_NAMES.items():
            prev = self.previous_counts[code]
            curr = current_counts[code]
            
            # Gem collected: count 1 -> 0 OR episode ended with gem gone
            # Key collected: count 2 -> 1 (board copy picked up, HUD copy remains)
            if code == 3:  # Gem
                # Check if gem was collected (count 1->0 or done with gem missing)
                if (prev == 1 and curr == 0) or (done and prev == 1 and curr != 1):
                    collected.append(name)
                    self.collection_order.append((step_number, name))
            else:  # Keys
                if prev == 2 and curr == 1:
                    collected.append(name)
                    self.collection_order.append((step_number, name))
        
        # Special case: If episode done and gem was present but now missing, it was collected
        if done and self.previous_counts[3] == 1:
            # Check if gem wasn't already recorded as collected
            if 'gem' not in [n for _, n in self.collection_order]:
                collected.append('gem')
                self.collection_order.append((step_number, 'gem'))
        
        self.previous_counts = current_counts
        return collected
    
    def get_collection_order(self) -> List[Tuple[int, str]]:
        """Get the order in which entities were collected."""
        return self.collection_order.copy()
    
    def get_collected_entities(self) -> List[str]:
        """Get list of all collected entity names."""
        return [name for _, name in self.collection_order]
    
    def _get_counts(self, state: EnvState) -> Dict[int, int]:
        """Get entity counts from state."""
        return {
            3: state.count_entities(9),        # gem (type 9)
            4: state.count_entities(2, 0),     # blue_key (type 2, theme 0)
            5: state.count_entities(2, 1),     # green_key (type 2, theme 1)
            6: state.count_entities(2, 2),     # red_key (type 2, theme 2)
        }