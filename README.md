# small

[简体中文](README.md) | [English](README_EN.md)

`small` 是一个极简的 Linux 单次任务执行 AI Agent。随着各类 AI CLI 的功能越来越丰富，工具本身也变得越来越复杂和沉重；但在很多场景中，我们只需要让 AI 快速完成一个临时任务。为此，`small` 将任务规划、命令执行和结果反馈集中在一个很小的二进制中，执行完当前任务后立即退出，不保留任何会话。

> **安全警告：** `small` 没有沙箱、权限隔离或强制操作审批机制。模型生成的命令会直接执行，并继承启动程序的当前用户权限；如果以 `root` 用户运行，命令将拥有完整的系统权限。错误或意外的命令可能造成数据丢失、系统损坏或其他严重后果，请仅在可信或隔离的环境中使用。

## 构建

```bash
make
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

模型会自行判断任务是否缺少必要信息或需要确认；需要时会在当前终端显示问题并等待输入，输入一行回答并按回车后，任务会在同一进程中继续执行。

## 开源协议

[Apache License 2.0](LICENSE)
