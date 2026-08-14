# small

## Build

```bash
CGO_ENABLED=0 go build -trimpath -ldflags="-s -w" -o small .
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
