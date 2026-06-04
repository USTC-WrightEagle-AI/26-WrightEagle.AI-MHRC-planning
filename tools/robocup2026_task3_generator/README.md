# RoboCup@Home 2026 Task 3 / GPSR Command Generator

本目录保存 RoboCup@Home 2026 GPSR 任务命令生成器的本地部署。

## 来源

- `CommandGenerator/`: https://github.com/RoboCupAtHome/CommandGenerator
- `CompetitionTemplate/`: https://github.com/RoboCupAtHome/CompetitionTemplate

RoboCup@Home 2026 RuleBook 的 GPSR 章节说明任务使用官方 CommandGenerator 生成。

## 已验证环境

- Conda 环境: `task3`
- Python: `3.10.20`
- 已安装包入口:
  - `athome-generator`
  - `athome-generator-gpsr-ui`

## 使用

打印配置:

```bash
./run_cli.sh --print-config
```

交互式生成命令:

```bash
./run_cli.sh
```

启动 GPSR UI:

```bash
./run_ui.sh
```

如果需要指定 OpenAI-compatible LLM:

```bash
./run_ui.sh --url http://127.0.0.1:11434/v1/chat/completions --api-key ollama --model qwen3:8b
```

## 本地补丁

上游当前版本在本机测试时有两个小问题，已在本地源码中修复:

1. `llm.py` 中 f-string 嵌套双引号导致 `SyntaxError`。
2. UI 默认写日志到 `~/gpsr-ui.log`，当前环境下不可写；改为默认写到运行目录的 `gpsr-ui.log`，也可用 `GPSR_UI_LOG` 环境变量覆盖。

