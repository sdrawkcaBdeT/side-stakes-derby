import sys
import os

# Get the path to the current script's directory (side-stakes-derby/)
current_dir = os.path.dirname(os.path.abspath(__file__))
# Get the path to the parent directory (the one containing both projects)
parent_dir = os.path.dirname(current_dir)
# Add the path to the 'prettyDerbyClubAnalysis' project to sys.path
other_project_path = os.path.join(parent_dir, 'prettyDerbyClubAnalysis')
sys.path.append(other_project_path)