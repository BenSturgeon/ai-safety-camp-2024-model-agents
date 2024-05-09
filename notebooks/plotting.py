import matplotlib.pyplot as plt

def plot_class_precision_with_title(class_precision_dict, title):
    # Enhanced colors for a more solid look
    enhanced_colors = {
        'top_left': '#ffd700',  # Gold for "top_left"
        'top_right': '#007bff',  # Strong blue
        'bottom_left': '#28a745',  # Strong green
        'bottom_right': '#dc3545',  # Strong red
    }

    # Plotting with enhanced colors
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(class_precision_dict.keys(), class_precision_dict.values(), color=[enhanced_colors.get(x, 'gray') for x in class_precision_dict.keys()])

    ax.set_xlabel('Class')
    ax.set_ylabel('Precision')
    ax.set_title(title)
    plt.xticks(rotation=45)  # Rotate class labels for better visibility

    # Adding the precision values on top of the bars
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, yval, round(yval, 2), va='bottom', ha='center')  # Align center horizontally

    plt.show()



def plot_objective_probe_data(class_precision_dict):
    # Colors for direction-related classes
    enhanced_colors = {
        'gem': '#ffd700',  # Gold for "gem"
        'blue_key': '#007bff',  # Strong blue
        'green_key': '#28a745',  # Strong green
        'red_key': '#dc3545',  # Strong red
        'blue_lock': '#0056b3',  # Deeper blue
        'green_lock': '#1e7e34',  # Deeper green
        'red_lock': '#b31b1b'  # Deeper red
    }

    # Plotting with specific colors
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(class_precision_dict.keys(), class_precision_dict.values(), color=[enhanced_colors.get(x, 'gray') for x in class_precision_dict.keys()])

    ax.set_xlabel('Class')
    ax.set_ylabel('Precision')
    ax.set_title('Objective Probe Data Logits')
    plt.xticks(rotation=45)  # Rotate class labels for better visibility

    # Adding the precision values on top of the bars
    for bar in bars:
        yval = bar.get_height()
        ax
def convert_report_to_dict(report_string):
    lines = report_string.strip().split('\n')
    class_precision_dict = {}
    for line in lines[2:-5]:  # Skip the header and summary lines
        parts = line.split(maxsplit=1)  # Split only on the first whitespace
        if parts:  # Check if the line was successfully split
            class_name = parts[0]
            precision_value = float(parts[1].strip().split()[0])  # Extract the precision value
            class_precision_dict[class_name] = precision_value
    return class_precision_dict