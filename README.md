# small

## 构建

```bash
CGO_ENABLED=0 go build -trimpath -ldflags="-s -w" -o small .
```

## 配置

```bash
export OPENAI_API_KEY="你的 API Key"
export OPENAI_ENDPOINT="https://你的服务地址/v1/chat/completions"
export OPENAI_MODEL="模型名称"
```

也可以使用 `OPENAI_BASE_URL`，程序会自动补全 `/v1/chat/completions`：

```bash
export OPENAI_BASE_URL="https://你的服务地址/v1"
```

## 使用

```bash
./small 用户要求
```

例如：

```bash
./small 查看当前主机的内存占用
./small "找出当前目录中最大的 10 个文件，但不要删除"
```
