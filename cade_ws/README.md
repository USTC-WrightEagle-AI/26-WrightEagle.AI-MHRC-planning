# CADE - Cognitive Agent for Domestic Environment

**Architecture: Modular ROS Pub/Sub with YOLO-World Support**

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                      CADE System                            │
│                                                             │
│  ┌──────────┐    /asr     ┌──────────┐    /tts     ┌──────┐ │
│  │cade_voice│ ──────────> │cade_brain│ ──────────> │cade_ │ │
│  │  (ASR)   │             │          │             │voice │ │
│  │          │             │ ┌──────┐ │             │(TTS) │ │
│  │          │             │ │ LLM  │ │             │      │ │
│  │          │             │ │Core  │ │             │      │ │
│  │          │             │ ├──────┤ │             │      │ │
│  │          │             │ │Memory│ │             │      │ │
│  │          │             │ ├──────┤ │             │      │ │
│  │          │             │ │Skills│ │             │      │ │
│  │          │             │ └──────┘ │             │      │ │
│  └──────────┘             └────┬─────┘             └──────┘ │
│                                │                            │
│                     /cade/task_cmd                          │
│                                │                            │
│                                v                            │
│                          ┌──────────┐                       │
│                          │cade_vision│                      │
│                          │ (YOLO +  │                      │
│                          │RealSense)│                      │
│                          └──────────┘                       │
│                                │                            │
│                     /vision/detections_3d                   │
│                     /cade/task_status                       │
│                                │                            │
│                                v                            │
│                          ┌──────────┐                       │
│                          │cade_brain│                       │
│                          │(receives │                       │
│                          │ results) │                       │
│                          └──────────┘                       │
└─────────────────────────────────────────────────────────────┘
```

## Package Structure

```
cade_ws/
├── src/
│   ├── cade_brain/          # Brain Layer - LLM controller & world model
│   │   ├── scripts/
│   │   │   └── brain_node.py        # Main entry point
│   │   └── src/cade_brain/
│   │       ├── controller.py        # Central dispatcher
│   │       ├── memory.py            # Dynamic world model (empty on boot)
│   │       ├── robot_interface.py   # Abstract interface
│   │       ├── llm_core/            # LLM logic (from original brain/)
│   │       │   ├── llm_client.py
│   │       │   ├── prompts.py
│   │       │   ├── schemas.py
│   │       │   └── config.py
│   │       └── skills/              # Skill dispatchers (ROS Pub/Sub)
│   │           ├── base_skill.py
│   │           ├── vision_skill.py
│   │           └── nav_skill.py
│   │
│   ├── cade_vision/          # Vision Layer - Pure feature extraction
│   │   ├── scripts/
│   │   │   └── open_vision_node.py  # YOLO + RealSense detector
│   │   └── src/cade_vision/
│   │
│   └── cade_voice/           # Voice Layer - ASR & TTS
│       ├── scripts/
│       │   ├── voice_bridge_node.py  # Voice pipeline manager
│       │   ├── asr_node.py           # Speech recognition
│       │   ├── tts_node.py           # Text-to-speech
│       │   └── ...                   # Other ASR variants
│       ├── launch/
│       │   └── voice.launch
│       ├── models/
│       └── test_audio/
│
├── launch/
│   └── cade_full.launch      # Launch all nodes
├── start_cade.sh             # One-click startup script
├── .env.example              # Environment template
└── README.md                 # This file
```

## Core Design Principles

### 1. Absolute Isolation
- **cade_vision**: Only "see" and publish coordinates. NO chassis control, NO model mutation.
- **cade_brain**: Only decide and dispatch. NO camera access, NO direct hardware control.
- **cade_voice**: Only listen and speak. NO LLM logic, NO vision processing.

### 2. Zero Hardcoding
- World model (memory.py) starts **empty on boot**.
- All data is dynamically populated by vision callbacks.
- No preset coordinates, people, or objects.

### 3. ROS Topic Communication (JSON over std_msgs/String)

| Topic | Direction | Description |
|-------|-----------|-------------|
| `/asr` | voice -> brain | Speech recognition results |
| `/tts` | brain -> voice | Text-to-speech requests |
| `/cade/task_cmd` | brain -> vision/nav | Task commands (JSON) |
| `/cade/task_status` | vision/nav -> brain | Task results (JSON) |
| `/vision/detections_3d` | vision -> brain | 3D detection data (JSON) |

### 4. Task Command Format

```json
{
  "action": "find_object",
  "target": "apple",
  "room": "kitchen"
}
```

### 5. Task Status Format

```json
{
  "status": "SUCCESS",
  "result": {
    "name": "apple",
    "position_3d": [0.5, 0.3, 1.2],
    "confidence": 0.95
  }
}
```

## How to Start

### Prerequisites

1. ROS Noetic/Melodic installed
2. Python dependencies:
   ```bash
   pip install -r ../requirements.txt
   ```
3. Build the workspace:
   ```bash
   cd cade_ws
   catkin_make
   source devel/setup.bash
   ```
4. Configure environment:
   ```bash
   cp .env.example .env
   # Edit .env with your API keys
   ```

### Quick Start

```bash
# Terminal 1: Start ROS core
roscore

# Terminal 2: Start all CADE nodes
cd cade_ws
source devel/setup.bash
./start_cade.sh full

# Or start individual layers:
./start_cade.sh voice    # ASR + TTS only
./start_cade.sh vision   # Vision only
./start_cade.sh brain    # Brain only
```

### Manual Start

```bash
# Voice layer
roslaunch cade_voice voice.launch

# Vision node (separate terminal)
rosrun cade_vision open_vision_node.py --model yolov8x-worldv2.pt --device cuda

# Brain node (separate terminal)
rosrun cade_brain brain_node.py --mode default
```

## Key Changes from Original (v0.1.0)

| Old | New |
|-----|-----|
| `body/robot.py` with hardcoded data | `cade_brain/memory.py` empty-on-boot |
| Controller calls `self.robot.method()` | Controller calls `VisionSkill`/`NavSkill` (ROS Pub/Sub) |
| Vision logic mixed in brain | `cade_vision/open_vision_node.py` - pure extraction |
| `main.py` entry point | `cade_brain/scripts/brain_node.py` |
| Monolithic structure | 3 independent ROS packages |

## LLM Action Space

The brain supports 22 action types (see `schemas.py` for full definitions):
- **Navigation**: `goToLoc`
- **Person**: `findPrsInRoom`, `meetPrsAtBeac`, `countPrsInRoom`, etc. (15 total)
- **Object**: `findObjInRoom`, `bringMeObjFromPlcmt`, `countObjOnPlcmt`, etc. (6 total)

Actions are dispatched through `VisionSkill` (perception) or `NavSkill` (movement) via ROS.
