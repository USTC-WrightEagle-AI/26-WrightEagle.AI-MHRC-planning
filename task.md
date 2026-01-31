## 5.1 MHRC Framework: A General Multi-Robot Collaboration Architecture

Our team has developed MHRC (MultiHeterogeneous Robot Collaboration), a
decentralized framework that leverages LLMs for multi-robot task planning and
coordination[11]. MHRC models collaborative tasks as a Decentralized Partially
Observable Markov Decision Process (DEC-POMDP), enabling each robot to
make autonomous decisions based on local observations and inter-robot communication.

Within MHRC, each robot operates an integrated perceptionmemoryplanning
loop to support long-horizon decentralized collaboration. Local observations and
inter-robot natural language messages are unified into a dynamically updated
scene graph, which captures both environment structure and shared task-relevant
states. Task execution context is maintained through a memory mechanism that
records feedback, communication, and action histories, with recent and critical
information explicitly highlighted to guide decisionmaking. Based on the current
observation and memory states, each robot independently performs LLMbased
task planning with ChainofThought reasoning, generating action sequences
from a predefined action set. Execution feedback is continuously incorporated
to enable adaptive replanning, while robotspecific feedback channels allow heterogeneous platformssuch as mobile manipulators and fixed manipulators
to adjust their behaviors according to their distinct capabilities.


## 5.2 Single-Robot variant Implementation for RoboCup@Home
In the RoboCup@Home system, we adopt a single-robot variant of the MHRC
framework. The observation module collects information from navigation, perception, and manipulation components. The memory module records task execution history and feedback from the environment.

Based on these inputs, the planning module uses a large language model to
decompose natural language instructions into a sequence of predefined actions,
such as navigation, grasping, and placement. During execution, the system monitors feedback from lowerlevel modules and updates the plan when necessary.
This design allows the robot to execute multistep tasks while responding to
changes in execution state