import os
import pandas as pd
import numpy as np
import glob
import json
import argparse
from collections import defaultdict
from typing import Dict, List, Tuple
from sklearn.cluster import DBSCAN

ENTITY_CODE_DESCRIPTION = {
    3: "gem",
    4: "blue_key",
    5: "green_key",
    6: "red_key",
    7: "blue_lock",
    8: "green_lock",
    9: "red_lock"
}

class EntityActivationSpanAnalyzer:
    def __init__(self, run_dir: str, coverage_percentile: float = 98):
        """
        Initialize the analyzer for computing entity-channel activation spans from a single run.
        
        Args:
            run_dir: Directory containing intervention_reached_log CSV files for a single run
            coverage_percentile: Percentile of interventions to capture (default 98%)
        """
        self.run_dir = run_dir
        self.coverage_percentile = coverage_percentile
        self.intervention_data = []
        
    def load_intervention_logs(self):
        """Load all intervention_reached_log CSV files from the run directory."""
        csv_pattern = os.path.join(self.run_dir, "intervention_reached_log*.csv")
        csv_files = glob.glob(csv_pattern)
        
        print(f"Found {len(csv_files)} intervention log files to process")
        
        for csv_file in csv_files:
            try:
                df = pd.read_csv(csv_file)
                self.intervention_data.append(df)
                print(f"  Loaded {os.path.basename(csv_file)}: {len(df)} records")
            except Exception as e:
                print(f"  Error loading {csv_file}: {e}")
        
        if self.intervention_data:
            self.combined_df = pd.concat(self.intervention_data, ignore_index=True)
            print(f"Total intervention records loaded: {len(self.combined_df)}")
        else:
            raise ValueError("No intervention log files could be loaded")
    
    def find_activation_ranges(self, values: np.ndarray, min_cluster_size: int = 5) -> List[Tuple[float, float]]:
        """
        Find activation ranges using clustering to ignore outliers.
        Returns list of (min, max) tuples for each cluster.
        """
        if len(values) < min_cluster_size:
            if len(values) > 0:
                return [(float(values.min()), float(values.max()))]
            return []
        
        X = values.reshape(-1, 1)
        
        value_range = values.max() - values.min()
        
        if value_range == 0:
            return [(float(values[0]), float(values[0]))]
        
        eps = value_range * 0.05
        
        dbscan = DBSCAN(eps=eps, min_samples=min_cluster_size)
        labels = dbscan.fit_predict(X)
        
        ranges = []
        total_points = len(values)
        clustered_points = 0
        
        for cluster_id in set(labels):
            if cluster_id == -1:
                continue
            
            cluster_values = values[labels == cluster_id]
            clustered_points += len(cluster_values)
            
            lower_percentile = (100 - self.coverage_percentile) / 2
            upper_percentile = 100 - lower_percentile
            
            lower = np.percentile(cluster_values, lower_percentile)
            upper = np.percentile(cluster_values, upper_percentile)
            
            ranges.append((float(lower), float(upper)))
        
        coverage = clustered_points / total_points
        if coverage < 0.9 and len(ranges) == 0:
            lower_percentile = (100 - self.coverage_percentile) / 2
            upper_percentile = 100 - lower_percentile
            lower = np.percentile(values, lower_percentile)
            upper = np.percentile(values, upper_percentile)
            ranges.append((float(lower), float(upper)))
        
        ranges.sort(key=lambda x: x[0])
        
        merged_ranges = []
        for r in ranges:
            if merged_ranges and r[0] <= merged_ranges[-1][1]:
                merged_ranges[-1] = (merged_ranges[-1][0], max(merged_ranges[-1][1], r[1]))
            else:
                merged_ranges.append(r)
        
        return merged_ranges
    
    def compute_entity_channel_spans(self) -> Dict[str, Dict[int, List[Tuple[float, float]]]]:
        """
        Compute activation spans for each entity-channel pair.
        Returns nested dict: entity -> channel -> list of (min, max) ranges
        """
        entity_channel_spans = {}
        
        for entity_name in ENTITY_CODE_DESCRIPTION.values():
            entity_data = self.combined_df[self.combined_df['TargetEntityName'] == entity_name]
            
            if entity_data.empty:
                print(f"  No data found for entity: {entity_name}")
                continue
            
            print(f"\nProcessing entity: {entity_name}")
            
            channel_spans = {}
            for channel in entity_data['Channel'].unique():
                channel_data = entity_data[entity_data['Channel'] == channel]
                
                if len(channel_data) < 3:
                    continue
                
                intervention_values = channel_data['InterventionValue'].values
                
                ranges = self.find_activation_ranges(intervention_values)
                
                if ranges:
                    channel_spans[int(channel)] = ranges
                    print(f"  Channel {channel}: {len(ranges)} range(s) found")
            
            if channel_spans:
                entity_channel_spans[entity_name] = dict(sorted(channel_spans.items()))
        
        return entity_channel_spans
    
    def compute_span_statistics(self) -> Dict:
        """Compute statistics about the activation spans."""
        spans = self.compute_entity_channel_spans()
        
        stats = {
            'total_entities_with_spans': len(spans),
            'entity_details': {}
        }
        
        for entity, channels in spans.items():
            entity_stats = {
                'num_channels': len(channels),
                'channels': {}
            }
            
            for channel, ranges in channels.items():
                entity_data = self.combined_df[
                    (self.combined_df['TargetEntityName'] == entity) & 
                    (self.combined_df['Channel'] == channel)
                ]
                
                range_stats = []
                for range_min, range_max in ranges:
                    in_range = entity_data[
                        (entity_data['InterventionValue'] >= range_min) & 
                        (entity_data['InterventionValue'] <= range_max)
                    ]
                    coverage = len(in_range) / len(entity_data) if len(entity_data) > 0 else 0
                    
                    range_stats.append({
                        'min': range_min,
                        'max': range_max,
                        'width': range_max - range_min,
                        'coverage': coverage,
                        'num_interventions_in_range': len(in_range),
                        'total_interventions': len(entity_data)
                    })
                
                entity_stats['channels'][channel] = range_stats
            
            stats['entity_details'][entity] = entity_stats
        
        return stats
    
    def save_spans_to_json(self, output_path: str):
        """Save computed spans and statistics to a JSON file."""
        spans = self.compute_entity_channel_spans()
        stats = self.compute_span_statistics()
        
        spans_with_str_keys = {}
        for entity, channels in spans.items():
            spans_with_str_keys[entity] = {str(ch): ranges for ch, ranges in channels.items()}
        
        output_data = {
            'activation_spans': spans_with_str_keys,
            'statistics': stats,
            'metadata': {
                'run_dir': self.run_dir,
                'coverage_percentile': self.coverage_percentile,
                'num_files_processed': len(self.intervention_data),
                'total_records': len(self.combined_df) if hasattr(self, 'combined_df') else 0,
                'entities': list(ENTITY_CODE_DESCRIPTION.values()),
                'total_channels_analyzed': len(self.combined_df['Channel'].unique()) if hasattr(self, 'combined_df') else 0
            }
        }
        
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"\nSaved activation spans to {output_path}")
        return output_data
    
    def generate_summary_report(self) -> str:
        """Generate a text summary of the analysis."""
        spans = self.compute_entity_channel_spans()
        stats = self.compute_span_statistics()
        
        report = []
        report.append("=" * 80)
        report.append("ENTITY-CHANNEL ACTIVATION SPANS REPORT")
        report.append("=" * 80)
        report.append(f"\nRun Directory: {self.run_dir}")
        report.append(f"Coverage Percentile: {self.coverage_percentile}%")
        report.append(f"Total Records Analyzed: {len(self.combined_df)}")
        
        for entity in ENTITY_CODE_DESCRIPTION.values():
            if entity not in spans:
                continue
                
            report.append(f"\n{'-' * 60}")
            report.append(f"{entity.upper()}")
            report.append(f"{'-' * 60}")
            
            channels = spans[entity]
            report.append(f"Active Channels: {len(channels)}")
            
            for channel in sorted(channels.keys()):
                ranges = channels[channel]
                channel_stats = stats['entity_details'][entity]['channels'][channel]
                
                report.append(f"\n  Channel {channel}:")
                for i, (range_info, (range_min, range_max)) in enumerate(zip(channel_stats, ranges)):
                    report.append(f"    Range {i+1}: [{range_min:.6f}, {range_max:.6f}]")
                    report.append(f"      - Width: {range_info['width']:.6f}")
                    report.append(f"      - Coverage: {range_info['coverage']:.1%} ({range_info['num_interventions_in_range']}/{range_info['total_interventions']} interventions)")
        
        return "\n".join(report)


def compute_spans_for_directory(run_dir: str, coverage_percentile: float = 98, use_cache: bool = True) -> Dict:
    """
    Compute or load cached entity activation spans for a given run directory.
    
    Args:
        run_dir: Directory containing intervention_reached_log CSV files
        coverage_percentile: Percentage of interventions to capture (default 98%)
        use_cache: Whether to use cached results if available
    
    Returns:
        Dictionary containing activation_spans, statistics, and metadata
    """
    cache_path = os.path.join(run_dir, "entity_activation_spans.json")
    
    if use_cache and os.path.exists(cache_path):
        with open(cache_path, 'r') as f:
            return json.load(f)
    
    analyzer = EntityActivationSpanAnalyzer(run_dir, coverage_percentile)
    analyzer.load_intervention_logs()
    return analyzer.save_spans_to_json(cache_path)


def main():
    parser = argparse.ArgumentParser(description="Compute entity activation spans from intervention logs")
    parser.add_argument("--run_dir", type=str, 
                       default="quantitative_intervention_runs/quantitative_interventions_base/base_serial_sweep_64_256_20250709_204312/run_base_conv4a_val_0-17p406613_ents_gem_blue_key_green_key_red_key_blue_lock_green_lock_red_lock_20250713_123949",
                       help="Directory containing intervention_reached_log CSV files for a single run")
    parser.add_argument("--output_json", type=str,
                       default=None,
                       help="Output JSON file path. If not specified, saves to run_dir/entity_activation_spans.json")
    parser.add_argument("--coverage", type=float, default=98,
                       help="Percentage of interventions to capture (default: 98)")
    parser.add_argument("--print_report", action="store_true",
                       help="Print summary report to console")
    parser.add_argument("--use_cache", action="store_true",
                       help="Use cached JSON if it exists in the run directory")
    
    args = parser.parse_args()
    
    if args.output_json is None:
        output_json = os.path.join(args.run_dir, "entity_activation_spans.json")
    else:
        output_json = args.output_json
    
    if args.use_cache and os.path.exists(output_json):
        print(f"Using cached results from: {output_json}")
        with open(output_json, 'r') as f:
            output_data = json.load(f)
        print(f"Loaded cached results with {len(output_data['activation_spans'])} entities")
    else:
        print(f"Initializing analyzer with {args.coverage}% coverage target...")
        analyzer = EntityActivationSpanAnalyzer(args.run_dir, args.coverage)
        
        print(f"Loading intervention logs from: {args.run_dir}")
        analyzer.load_intervention_logs()
        
        print("\nComputing activation spans...")
        output_data = analyzer.save_spans_to_json(output_json)
    
    if args.print_report and not args.use_cache:
        analyzer = EntityActivationSpanAnalyzer(args.run_dir, args.coverage)
        analyzer.load_intervention_logs()
        report = analyzer.generate_summary_report()
        print("\n" + report)
    
    print("\n" + "=" * 60)
    print("QUICK SUMMARY")
    print("=" * 60)
    for entity, channels in output_data['activation_spans'].items():
        total_ranges = sum(len(ranges) for ranges in channels.values())
        print(f"{entity}: {len(channels)} channels, {total_ranges} total ranges")
    
    print(f"\nResults saved to: {output_json}")


if __name__ == "__main__":
    main()