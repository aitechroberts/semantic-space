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

After extraction, you should have (v0.0.2 release — **scene
directories live directly under the repo root, not under ``data/``**,
and the ground-truth file is ``ground_truth.json``):

```
Space3D-Bench/
├── room_0/
│   ├── questions.json        # Questions for this scene
│   ├── ground_truth.json     # Ground truth answers (NOT answers.json)
│   ├── img/                  # Reference images for VLM-judge questions
│   └── misc/                 # Curated 3D detections + extra assets
├── room_1/
├── office_2/
├── office_3/
└── ... (other scenes)
```

> Older drafts of this playbook referenced ``data/<scene>/answers.json``.
> That layout does not match the current release. The rest of the
> pipeline uses ``generate_groundtruth/_space3d_layout.py`` to resolve
> paths and schema consistently; if you touch any Space3D-Bench path
> directly, go through that helper.

## Step 4: Extract Questions for Your Scenes

```bash
# Check question counts per scene
for scene in room_0 room_1 office_2 office_3; do
    echo -n "$scene: "
    cat "${scene}/questions.json" | python3 -c "import sys,json; print(len(json.load(sys.stdin)))"
done
```

Expected output for v0.0.2:
```
room_0:   60 questions
room_1:   60 questions
office_2: 60 questions
office_3: 50 questions
(total: 230 across the four scenes we run)
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

### ground_truth.json

Each entry is `{"answer": <str|dict>, "prompt": <str>}`. There is
**no categorical `type` field** — the question taxonomy is encoded by
the `answer` format itself. Representative examples:

```json
{
  "1":  {"answer": "3D position: [[3.669, -1.103, 0.077]]",
         "prompt": "The answer should contain 3D positions... 0.1 m tolerance..."},
  "17": {"answer": "Number of objects: 2",
         "prompt": "The answer should contain a number of objects matching the ground truth."},
  "25": {"answer": "Objects: ['book', 'candle', 'book', 'vase']",
         "prompt": "The answer should contain a list of objects matching the ground truth..."},
  "32": {"answer": "From object=9 at [3.7, -0.52, -1.05] to object=77 at [3.75, 2.64, -1.04] the navigable distance is 2.52 meters.",
         "prompt": "The answer should specify the distance in meters... 0.5 m tolerance..."},
  "49": {"answer": {"image_path": "data/room_0/img/q49.png",
                    "example_answer": "Both sofas have the light color..."},
         "prompt": "You are provided with the RGB image, divided..."},
  "57": {"answer": "No",
         "prompt": "The answer should contain a clear 'yes' or 'no' response..."}
}
```

`_space3d_layout.infer_type` bins those into
`binary / object_list / qualitative / count / position / distance`. The
Phase-B deterministic scorer handles the first three; the remainder
are reported as `format_unsupported`.

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
# View questions for room_0
cat room_0/questions.json | python3 -m json.tool

# Count total questions across your scenes
total=0
for s in room_0 room_1 office_2 office_3; do
    n=$(cat "${s}/questions.json" | python3 -c "import sys,json; print(len(json.load(sys.stdin)))")
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
        with open(f'{scene}/questions.json') as qf:
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
