# Space3D-Bench Setup Playbook

## Overview

[Space3D-Bench](https://github.com/Space3D-Bench/Space3D-Bench) is a spatial 3D question answering benchmark containing 1000 questions across 13 Replica dataset scenes.

**Relevant scenes for your evaluation:**
- `room0`, `room1` - apartment rooms
- `office2`, `office3` - office rooms

---

## Step 1: Clone the Repository

```bash
cd ~/cmu-grad  # or your preferred directory
git clone https://github.com/Space3D-Bench/Space3D-Bench.git
cd Space3D-Bench
```

## Step 2: Download the Data

```bash
# Download the data archive (contains questions, answers, and curated detections)
wget https://github.com/Space3D-Bench/Space3D-Bench/releases/download/v0.0.2/data.zip

# Extract to the repository
unzip data.zip -d .
rm data.zip
```

## Step 3: Verify Structure

After extraction, you should have:

```
Space3D-Bench/
├── data/
│   ├── room_0/
│   │   ├── questions.json        # Questions for this scene
│   │   ├── answers.json          # Ground truth answers
│   │   └── detections/           # Curated 3D detections
│   ├── room_1/
│   ├── office_2/
│   ├── office_3/
│   └── ... (other scenes)
├── assessment/                    # Evaluation scripts
└── README.md
```

## Step 4: Extract Questions for Your Scenes

```bash
# Check question counts per scene
for scene in room_0 room_1 office_2 office_3; do
    echo -n "$scene: "
    cat "data/${scene}/questions.json" | python3 -c "import sys,json; print(len(json.load(sys.stdin)))"
done
```

Expected output (approximately):
```
room_0: ~80 questions
room_1: ~75 questions  
office_2: ~85 questions
office_3: ~70 questions
```

## Step 5: Understand the Data Format

### questions.json
```json
{
    "1": "How many chairs are there in the room?",
    "2": "What is the color of the sofa?",
    "3": "Is the lamp closer to the desk or the bed?"
}
```

### answers.json
```json
{
    "1": {
        "answer": "There are 3 chairs.",
        "type": "count",
        "acceptance_criterion": "exact_match"
    },
    "2": {
        "answer": "The sofa is blue.",
        "type": "attribute",
        "acceptance_criterion": "semantic_match"
    }
}
```

---

## Step 6: Scene Name Mapping

**Important:** Space3D-Bench uses underscores, your data uses no separators:

| Space3D-Bench | Your Data |
|---------------|-----------|
| `room_0` | `room0` |
| `room_1` | `room1` |
| `office_2` | `office2` |
| `office_3` | `office3` |

The VQA scripts handle this mapping automatically.

---

## Step 7: Install Dependencies for Assessment

```bash
# For the official Space3D-Bench assessment system
pip install openai transformers torch

# For our custom VQA evaluation
pip install torch transformers accelerate
pip install qwen-vl-utils  # For Qwen3-VL
```

---

## Quick Reference Commands

```bash
# View questions for room0
cat data/room_0/questions.json | python3 -m json.tool

# Count total questions across your scenes
total=0
for s in room_0 room_1 office_2 office_3; do
    n=$(cat "data/${s}/questions.json" | python3 -c "import sys,json; print(len(json.load(sys.stdin)))")
    total=$((total + n))
    echo "$s: $n questions"
done
echo "Total: $total questions"

# Export questions to CSV for review
python3 -c "
import json
import csv

scenes = ['room_0', 'room_1', 'office_2', 'office_3']
with open('all_questions.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['scene', 'q_id', 'question'])
    for scene in scenes:
        with open(f'data/{scene}/questions.json') as qf:
            questions = json.load(qf)
            for qid, q in questions.items():
                writer.writerow([scene, qid, q])
print('Exported to all_questions.csv')
"
```

---

## Next Steps

1. Run the VQA evaluation using `run_vqa_eval.py`
2. Use `vlm_load_and_query.py` to query individual models
3. Use `clip_load_and_query.py` for CLIP-based retrieval experiments

See the main evaluation scripts for detailed usage.
