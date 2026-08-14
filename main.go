package main

import (
	"context"
	"errors"
	"fmt"
	"net"
	"net/http"
	"os"
	"os/signal"
	"strings"
	"syscall"
	"time"
)

const maxTaskDuration = 20 * time.Minute

func main() {
	os.Exit(run())
}

func run() int {
	task := strings.TrimSpace(strings.Join(os.Args[1:], " "))
	if task == "" {
		fmt.Fprintln(os.Stderr, "错误：请提供要执行的任务")
		fmt.Fprintln(os.Stderr, "用法：./small <用户要求>")
		return 2
	}

	config, err := loadConfig()
	if err != nil {
		fmt.Fprintln(os.Stderr, "配置错误：", err)
		return 2
	}

	cwd, err := os.Getwd()
	if err != nil {
		fmt.Fprintln(os.Stderr, "读取当前目录失败：", err)
		return 1
	}

	signalCtx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()
	ctx, cancel := context.WithTimeout(signalCtx, maxTaskDuration)
	defer cancel()

	httpClient := &http.Client{
		Transport: &http.Transport{
			Proxy: http.ProxyFromEnvironment,
			DialContext: (&net.Dialer{
				Timeout:   10 * time.Second,
				KeepAlive: 30 * time.Second,
			}).DialContext,
			ForceAttemptHTTP2:     true,
			MaxIdleConns:          10,
			IdleConnTimeout:       90 * time.Second,
			TLSHandshakeTimeout:   10 * time.Second,
			ExpectContinueTimeout: time.Second,
		},
	}

	client := &OpenAIClient{
		apiKey:       config.APIKey,
		endpoint:     config.Endpoint,
		model:        config.Model,
		instructions: buildInstructions(cwd, time.Now()),
		httpClient:   httpClient,
	}
	agent := &Agent{
		client:   client,
		executor: &CommandExecutor{cwd: cwd},
		stdout:   os.Stdout,
		stderr:   os.Stderr,
	}

	if err := agent.Run(ctx, task); err != nil {
		switch {
		case errors.Is(err, context.Canceled):
			fmt.Fprintln(os.Stderr, "\n任务已取消")
			return 130
		case errors.Is(err, context.DeadlineExceeded):
			fmt.Fprintln(os.Stderr, "\n任务超时")
			return 124
		default:
			fmt.Fprintln(os.Stderr, "\n执行失败：", err)
			return 1
		}
	}
	return 0
}
