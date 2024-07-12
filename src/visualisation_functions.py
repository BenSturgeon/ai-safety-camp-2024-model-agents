def plot_top_f1_scores(new_classification_data, categories, colors, top_n=15, title=None):
    import matplotlib.pyplot as plt
    import numpy as np

    # Parse f1-scores from the classification reports
    f1_scores = [report['0']['f1-score'] for report in new_classification_data]

    # Sort categories and scores by F1-score in descending order
    sorted_data = sorted(zip(categories, f1_scores, colors), key=lambda x: x[1], reverse=True)
    top_data = sorted_data[:top_n]  # Get top N scores

    # Unpack the sorted data
    top_categories, top_f1_scores, top_colors = zip(*top_data)

    # Print F1-scores for top N
    print(f"Top {top_n} F1-scores:")
    for category, score in zip(top_categories, top_f1_scores):
        print(f"{category}: {score:.4f}")

    # Create bar plot with zoomed-in y-axis for top N scores
    plt.figure(figsize=(12, 6))
    bars = plt.bar(top_categories, top_f1_scores, color=top_colors)
    if title:
        plt.title(title)
    plt.xlabel('Categories')
    plt.ylabel('F1-score')
    plt.ylim(max(0.5, min(top_f1_scores) - 0.05), 1.0)  # Dynamically set lower y-limit
    plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels for better readability

    # Add value labels on top of each bar
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                 f'{height:.3f}',
                 ha='center', va='bottom', rotation=0)

    plt.tight_layout()
    plt.show()