#!/usr/bin/env python3
"""Utility functions for detecting and managing HUD keys in Heist environments."""

def detect_hud_keys(state):
    """
    Detect keys in the HUD using entity state.
    
    Args:
        state: EnvState from heist.state_from_venv()
    
    Returns:
        dict: {
            'count': number of keys in HUD,
            'positions': list of (x, y) tuples for each HUD key
        }
    """
    hud_keys = []
    
    for entity in state.get_entities():
        entity_type = entity['type'].val
        
        # Type 11 is KEY_ON_RING (HUD keys)
        if entity_type == 11:
            y_pos = entity['y'].val
            
            # HUD is at top of screen (y < 0.5 in Heist coordinates)
            # Natural maze HUD is at y ≈ 0.02
            if y_pos < 0.5:
                x_pos = entity['x'].val
                hud_keys.append((x_pos, y_pos))
    
    return {
        'count': len(hud_keys),
        'positions': hud_keys
    }


def verify_hud_working(venv):
    """
    Verify if HUD is properly functioning in an environment.
    
    Args:
        venv: The environment to check
    
    Returns:
        bool: True if HUD keys are present and visible
    """
    from utils import heist
    
    state = heist.state_from_venv(venv, 0)
    hud_info = detect_hud_keys(state)
    
    # Check if there are any type 11 entities (KEY_ON_RING)
    has_type_11 = False
    for entity in state.get_entities():
        if entity['type'].val == 11:
            has_type_11 = True
            break
    
    # HUD is working if we have type 11 entities in HUD position
    return hud_info['count'] > 0 and has_type_11