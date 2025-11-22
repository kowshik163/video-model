import os
import json
import argparse
import matplotlib.pyplot as plt
import pandas as pd
import glob

def analyze_session(log_dir):
    """
    Analyzes a single video session logs to correlate transitions, beats, and routing.
    """
    print(f"Analyzing session in: {log_dir}")
    
    # Load Data
    transitions = load_jsonl(os.path.join(log_dir, "visual", "transitions.jsonl"))
    beats = load_jsonl(os.path.join(log_dir, "visual", "audio_beat.jsonl"))
    decisions = load_jsonl(os.path.join(log_dir, "supervisor", "decisions.jsonl"))
    
    if not decisions:
        print("No supervisor decisions found.")
        return

    # Convert to DataFrame
    df_decisions = pd.DataFrame(decisions)
    
    # Plotting
    plt.figure(figsize=(12, 6))
    
    # 1. Consistency Score
    plt.plot(df_decisions['timestamp'], df_decisions['consistency'], label='Consistency Score', color='blue', alpha=0.6)
    
    # 2. Routing Decisions (Scatter)
    # Filter where needs_3d is True
    triggers = df_decisions[df_decisions['needs_3d'] == True]
    plt.scatter(triggers['timestamp'], triggers['consistency'], color='red', label='3D Trigger', zorder=5)
    
    # 3. Transitions (Vertical Lines)
    for t in transitions:
        plt.axvline(x=t['timestamp'], color='green', linestyle='--', alpha=0.5, label='Visual Transition' if 'Visual Transition' not in plt.gca().get_legend_handles_labels()[1] else "")
        
    # 4. Audio Beats (Small ticks)
    for b in beats:
        plt.plot(b['timestamp'], 0, marker='|', color='purple', markersize=10, alpha=0.3, label='Audio Beat' if 'Audio Beat' not in plt.gca().get_legend_handles_labels()[1] else "")

    plt.title(f"Session Analysis: {os.path.basename(log_dir)}")
    plt.xlabel("Time (s)")
    plt.ylabel("Consistency / Complexity")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    output_plot = os.path.join(log_dir, "session_analysis.png")
    plt.savefig(output_plot)
    print(f"Saved analysis plot to {output_plot}")
    
    # Text Summary
    print("\n--- Analysis Summary ---")
    print(f"Total Frames: {len(df_decisions)}")
    print(f"3D Triggers: {len(triggers)} ({len(triggers)/len(df_decisions)*100:.1f}%)")
    print(f"Visual Transitions Detected: {len(transitions)}")
    print(f"Audio Beats Detected: {len(beats)}")
    
    # Correlation Check
    # Check how many triggers are within 0.5s of a transition
    correlated = 0
    for _, row in triggers.iterrows():
        t = row['timestamp']
        # Check transitions
        is_near = False
        for trans in transitions:
            if abs(trans['timestamp'] - t) < 0.5:
                is_near = True
                break
        if is_near:
            correlated += 1
            
    print(f"Triggers near Transitions: {correlated} ({correlated/len(triggers)*100 if len(triggers) else 0:.1f}%)")

def load_jsonl(path):
    data = []
    if os.path.exists(path):
        with open(path, 'r') as f:
            for line in f:
                try:
                    data.append(json.loads(line))
                except:
                    pass
    return data

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("log_dir", type=str, help="Path to specific video log directory (e.g. eval_logs/video1)")
    args = parser.parse_args()
    
    analyze_session(args.log_dir)
