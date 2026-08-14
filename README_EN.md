# small

[简体中文](README.md) | [English](README_EN.md)

`small` is a minimal AI agent for one-off Linux tasks. As AI CLI tools gain more features, they also become increasingly complex and heavyweight, while many situations only require AI to complete a single temporary task. `small` keeps task planning, command execution, and result reporting in one small binary, then exits without preserving a session.

> **Security warning:** `small` has no sandbox, permission isolation, or mandatory operation approval mechanism. Model-generated commands are executed directly with the privileges of the current user. When run as `root`, commands have unrestricted system access. Incorrect or unexpected commands may cause data loss, system damage, or other serious consequences. Use it only in trusted or isolated environments.

## Build

```bash
make
```

## Configuration

```bash
export OPENAI_API_KEY="your API key"
export OPENAI_ENDPOINT="https://your-service.example/v1/chat/completions"
export OPENAI_MODEL="your-model-name"
```

You can also use `OPENAI_BASE_URL`. The program will automatically append `/v1/chat/completions`:

```bash
export OPENAI_BASE_URL="https://your-service.example/v1"
```

## Usage

```bash
./small your request
```

Examples:

```bash
./small check the current system memory usage
./small "find the 10 largest files in the current directory, but do not delete anything"
```

When required information or confirmation is missing, the program displays a question in the current terminal and waits for one line of input. You can also enter a follow-up after the model returns a final response, or press Enter to finish.

## License

[Apache License 2.0](LICENSE)
