import helpers
import heist
import random
import numpy as np
import imageio

ENTITY_COLORS = {
    "blue": 0,
    "green": 1,
    "red": 2
}

ENTITY_TYPES = {
    "key": 2,
    "lock": 1,
    "gem": 9,
    "player": 0
}

ordered_layer_names  = {
    1: 'conv1a',
    2: 'pool1',
    3: 'conv2a',
    4: 'conv2b',
    5: 'pool2',
    6: 'conv3a',
    7: 'pool3',
    8: 'conv4a',
    9: 'pool4',
    10: 'fc1',
    11: 'fc2',
    12: 'fc3',
    13: 'value_fc',
    14: 'dropout_conv',
    15: 'dropout_fc'
}


def run_entity_steering_experiment_by_channel(model_path, layer_number, modification_value, episode, channel, entity_name, entity_color=None, num_levels=1, start_level=5, episode_timeout=200, save_gif=False):
    entity_type = ENTITY_TYPES.get(entity_name)
    entity_theme = ENTITY_COLORS.get(entity_color) if entity_color else None
    if entity_type is None:
        print(f"Invalid entity name: {entity_name}")
        return None

    venv = heist.create_venv(num=1, num_levels=num_levels, start_level=start_level)
    state = heist.state_from_venv(venv, 0)
    while not state.entity_exists(entity_type, entity_theme):
        start_level = random.randint(1, 100000)
        venv = heist.create_venv(num=1, num_levels=num_levels, start_level=start_level)
        state = heist.state_from_venv(venv, 0)
    print(state.entity_exists(entity_type, entity_theme))
    unchanged_obs = venv.reset()

    # Save the current position of the target entity
    original_position = state.get_entity_position(entity_type, entity_theme)

    # Move the target entity off-screen
    state.remove_entity(entity_type, entity_theme)

    state_bytes = state.state_bytes
    if state_bytes is not None:
        venv.env.callmethod("set_state", [state_bytes])
        modified_obs = venv.reset()

    state = heist.state_from_venv(venv, 0)

    # Restore the entity to its original position
    state.restore_entity_position(entity_type, entity_theme, original_position)

    state_bytes = state.state_bytes
    if state_bytes is not None:
        venv.env.callmethod("set_state", [state_bytes])

    model = helpers.load_interpretable_model(model_path=model_path)
    steering_layer_unchanged = ordered_layer_names[layer_number]
    steering_layer = helpers.rename_path(steering_layer_unchanged)

    model_activations = helpers.ModelActivations(model)
    model_activations.clear_hooks()
    output1, unmodified_activations = model_activations.run_with_cache(helpers.observation_to_rgb(unchanged_obs), [ordered_layer_names[layer_number]])
    model_activations.clear_hooks()
    output2, modified_obs_activations = model_activations.run_with_cache(helpers.observation_to_rgb(modified_obs), [ordered_layer_names[layer_number]])

    steering_vector = unmodified_activations[steering_layer][0][channel] - modified_obs_activations[steering_layer][0][channel]


    observation = venv.reset()
    done = False
    total_reward = 0
    frames = []
    observations = []
    count = 0
    entity_picked_up = False
    count_pickups = 0
    steps_until_pickup = 0
    
    # Count initial number of target entities
    initial_state = heist.state_from_venv(venv, 0)
    initial_entity_count = initial_state.count_entities(entity_type, entity_theme)
    
    while not done:
        if save_gif:
            frames.append(venv.render(mode='rgb_array'))
        
        observation = np.squeeze(observation)
        observation = np.transpose(observation, (1, 2, 0))
        converted_obs = helpers.observation_to_rgb(observation)
        action = helpers.generate_action_with_steering(model, converted_obs, steering_vector, steering_layer, modification_value, is_procgen_env=True)

        observation, reward, done, info = venv.step(action)
        total_reward += reward
        observations.append(converted_obs)
        steps_until_pickup += 1
        if steps_until_pickup > 300:
            done = True
        
        state = heist.state_from_venv(venv, 0)
        current_entity_count = state.count_entities(entity_type, entity_theme)

        if current_entity_count < initial_entity_count:
            entity_picked_up = True
            done = True
            count_pickups += 1
            print(f"{entity_name.capitalize()} picked up after {steps_until_pickup} steps")
        elif entity_type == ENTITY_TYPES['gem'] and reward > 0:
            entity_picked_up = True
            done = True
            count_pickups += 1
            print(f"{entity_name.capitalize()} picked up after {steps_until_pickup} steps")

    if save_gif:
        imageio.mimsave(f'episode_steering_{episode}_green__key_layer_30.gif', frames, fps=30)
        print("Saved gif!")

    if not entity_picked_up:
        print(f"{entity_name.capitalize()} was not picked up during the episode")
    else:
        print(f"{entity_name.capitalize()} picked up during the episode")

    state = heist.state_from_venv(venv, 0)
    state_vals = state.state_vals

    lock_positions_after = heist.get_lock_statuses(state_vals)

    return total_reward, steps_until_pickup, count_pickups

def run_entity_steering_experiment(model_path, layer_number, modification_value, episode, entity_name, entity_color=None, num_levels=1, start_level=5, episode_timeout=200, save_gif=False):
    entity_type = ENTITY_TYPES.get(entity_name)
    entity_theme = ENTITY_COLORS.get(entity_color) if entity_color else None
    if entity_type is None:
        print(f"Invalid entity name: {entity_name}")
        return None

    venv = heist.create_venv(num=1, num_levels=num_levels, start_level=start_level)
    state = heist.state_from_venv(venv, 0)
    while not state.entity_exists(entity_type, entity_theme):
        start_level = random.randint(1, 100000)
        venv = heist.create_venv(num=1, num_levels=num_levels, start_level=start_level)
        state = heist.state_from_venv(venv, 0)
    print(state.entity_exists(entity_type, entity_theme))
    unchanged_obs = venv.reset()

    # Save the current position of the target entity
    original_position = state.get_entity_position(entity_type, entity_theme)

    # Move the target entity off-screen
    state.remove_entity(entity_type, entity_theme)

    state_bytes = state.state_bytes
    if state_bytes is not None:
        venv.env.callmethod("set_state", [state_bytes])
        modified_obs = venv.reset()

    state = heist.state_from_venv(venv, 0)

    # Restore the entity to its original position
    state.restore_entity_position(entity_type, entity_theme, original_position)

    state_bytes = state.state_bytes
    if state_bytes is not None:
        venv.env.callmethod("set_state", [state_bytes])

    model = helpers.load_interpretable_model(model_path=model_path)
    steering_layer_unchanged = ordered_layer_names[layer_number]
    steering_layer = helpers.rename_path(steering_layer_unchanged)

    model_activations = helpers.ModelActivations(model)
    model_activations.clear_hooks()
    output1, unmodified_activations = model_activations.run_with_cache(helpers.observation_to_rgb(unchanged_obs), [ordered_layer_names[layer_number]])
    model_activations.clear_hooks()
    output2, modified_obs_activations = model_activations.run_with_cache(helpers.observation_to_rgb(modified_obs), [ordered_layer_names[layer_number]])

    steering_vector = unmodified_activations[steering_layer][0] - modified_obs_activations[steering_layer][0]

    observation = venv.reset()
    done = False
    total_reward = 0
    frames = []
    observations = []
    count = 0
    entity_picked_up = False
    count_pickups = 0
    steps_until_pickup = 0
    
    # Count initial number of target entities
    initial_state = heist.state_from_venv(venv, 0)
    initial_entity_count = initial_state.count_entities(entity_type, entity_theme)
    
    while not done:
        if save_gif:
            frames.append(venv.render(mode='rgb_array'))
        
        observation = np.squeeze(observation)
        observation = np.transpose(observation, (1, 2, 0))
        converted_obs = helpers.observation_to_rgb(observation)
        action = helpers.generate_action_with_steering(model, converted_obs, steering_vector, steering_layer, modification_value, is_procgen_env=True)

        observation, reward, done, info = venv.step(action)
        total_reward += reward
        observations.append(converted_obs)
        steps_until_pickup += 1
        if steps_until_pickup > 300:
            done = True
        
        state = heist.state_from_venv(venv, 0)
        current_entity_count = state.count_entities(entity_type, entity_theme)

        if current_entity_count < initial_entity_count:
            entity_picked_up = True
            done = True
            count_pickups += 1
            print(f"{entity_name.capitalize()} picked up after {steps_until_pickup} steps")
        elif entity_type == ENTITY_TYPES['gem'] and reward > 0:
            entity_picked_up = True
            done = True
            count_pickups += 1
            print(f"{entity_name.capitalize()} picked up after {steps_until_pickup} steps")

    if save_gif:
        imageio.mimsave(f'episode_steering_{episode}.gif', frames, fps=30)
        print("Saved gif!")

    if not entity_picked_up:
        print(f"{entity_name.capitalize()} was not picked up during the episode")
    else:
        print(f"{entity_name.capitalize()} picked up during the episode")

    state = heist.state_from_venv(venv, 0)
    state_vals = state.state_vals

    lock_positions_after = heist.get_lock_statuses(state_vals)

    return total_reward, steps_until_pickup, count_pickups
def steering_vector(venv, model_path, layer_number, steering_channel, modification_value, episode, entity_name, entity_color=None, num_levels=0, start_level=5, episode_timeout=200, save_gif=False, filepath= '../gifs/unsteered_run.gif'):
    model = helpers.load_interpretable_model(model_path=model_path)
    original_env = venv
    state = heist.state_from_venv(venv, 0)
    env_state = heist.EnvState(state.state_bytes)
    entity_type = ENTITY_TYPES.get(entity_name)
    entity_theme = ENTITY_COLORS.get(entity_color) if entity_color else None
    print(entity_theme)
    original_position = env_state.get_key_position(entity_theme)
    unchanged_obs = venv.reset()
    unsteered_actions = helpers.run_episode_and_save_as_gif(env=original_env,model=model, save_gif=save_gif, filepath=filepath + "unsteered.gif")


    if entity_type is None:
        print(f"Invalid entity name: {entity_name}")
        return None
    
    # Move the target entity off-screen
    state.move_entity_offscreen(entity_type, entity_theme)

    state_bytes = state.state_bytes
    if state_bytes is not None:
        venv.env.callmethod("set_state", [state_bytes])
        modified_obs = venv.reset()

    state = heist.state_from_venv(venv, 0)

    # Restore the entity to its original position
    state.restore_entity_position(entity_type, entity_theme)

    state_bytes = state.state_bytes
    if state_bytes is not None:
        venv.env.callmethod("set_state", [state_bytes])

    
    steering_layer_unchanged = ordered_layer_names[layer_number]
    steering_layer = helpers.rename_path(steering_layer_unchanged)

    model_activations = helpers.ModelActivations(model)
    model_activations.clear_hooks()
    output1, unmodified_activations = model_activations.run_with_cache(helpers.observation_to_rgb(unchanged_obs), [ordered_layer_names[layer_number]])
    model_activations.clear_hooks()
    output2, modified_obs_activations = model_activations.run_with_cache(helpers.observation_to_rgb(modified_obs), [ordered_layer_names[layer_number]])

    steering_vector = unmodified_activations[steering_layer][0] - modified_obs_activations[steering_layer][0]

    actions = helpers.run_episode_with_steering_channels_and_save_as_gif(
        venv, model, steering_vector, steering_layer=ordered_layer_names[layer_number],
        steering_channel = steering_channel, modification_value=modification_value, filepath=filepath + "steered.gif",
        save_gif=save_gif, episode_timeout=episode_timeout
    )

    return steering_vector, original_position
