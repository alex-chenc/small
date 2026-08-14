package main

import (
	"context"
	"errors"
	"fmt"
	"io"
	"os"
	"os/exec"
	"strings"
	"sync"
	"syscall"
	"time"
)

const (
	defaultCommandTimeout = 60 * time.Second
	maxCommandTimeout     = 5 * time.Minute
	maxCapturedBytes      = 64 << 10
)

type CommandResult struct {
	ExitCode        int    `json:"exit_code"`
	Stdout          string `json:"stdout"`
	Stderr          string `json:"stderr"`
	DurationMS      int64  `json:"duration_ms"`
	TimedOut        bool   `json:"timed_out"`
	OutputTruncated bool   `json:"output_truncated"`
}

type CommandExecutor struct {
	cwd string
}

func (e *CommandExecutor) Execute(ctx context.Context, command string, requestedTimeout time.Duration, stdout, stderr io.Writer) CommandResult {
	timeout := requestedTimeout
	if timeout <= 0 {
		timeout = defaultCommandTimeout
	}
	if timeout > maxCommandTimeout {
		timeout = maxCommandTimeout
	}

	commandCtx, cancel := context.WithTimeout(ctx, timeout)
	defer cancel()

	started := time.Now()
	stdoutCapture := newCappedBuffer(maxCapturedBytes)
	stderrCapture := newCappedBuffer(maxCapturedBytes)

	cmd := exec.Command("/bin/bash", "-lc", command)
	cmd.Dir = e.cwd
	cmd.Env = commandEnvironment(os.Environ())
	cmd.SysProcAttr = &syscall.SysProcAttr{Setpgid: true, Pdeathsig: syscall.SIGKILL}

	stdoutPipe, err := cmd.StdoutPipe()
	if err != nil {
		return commandStartFailure(started, fmt.Errorf("创建 stdout pipe：%w", err))
	}
	stderrPipe, err := cmd.StderrPipe()
	if err != nil {
		return commandStartFailure(started, fmt.Errorf("创建 stderr pipe：%w", err))
	}
	if err := cmd.Start(); err != nil {
		return commandStartFailure(started, fmt.Errorf("启动命令：%w", err))
	}

	var copies sync.WaitGroup
	copies.Add(2)
	go func() {
		defer copies.Done()
		_, _ = io.Copy(io.MultiWriter(stdout, stdoutCapture), stdoutPipe)
	}()
	go func() {
		defer copies.Done()
		_, _ = io.Copy(io.MultiWriter(stderr, stderrCapture), stderrPipe)
	}()

	waitCh := make(chan error, 1)
	go func() { waitCh <- cmd.Wait() }()

	var waitErr error
	timedOut := false
	select {
	case waitErr = <-waitCh:
	case <-commandCtx.Done():
		timedOut = errors.Is(commandCtx.Err(), context.DeadlineExceeded)
		_ = syscall.Kill(-cmd.Process.Pid, syscall.SIGTERM)
		select {
		case waitErr = <-waitCh:
		case <-time.After(500 * time.Millisecond):
			_ = syscall.Kill(-cmd.Process.Pid, syscall.SIGKILL)
			waitErr = <-waitCh
		}
	}
	copies.Wait()

	exitCode := 0
	if timedOut {
		exitCode = 124
	} else if errors.Is(commandCtx.Err(), context.Canceled) {
		exitCode = 130
	} else if waitErr != nil {
		var exitErr *exec.ExitError
		if errors.As(waitErr, &exitErr) {
			exitCode = exitErr.ExitCode()
		} else {
			exitCode = -1
		}
	}

	return CommandResult{
		ExitCode:        exitCode,
		Stdout:          stdoutCapture.String(),
		Stderr:          stderrCapture.String(),
		DurationMS:      time.Since(started).Milliseconds(),
		TimedOut:        timedOut,
		OutputTruncated: stdoutCapture.Truncated() || stderrCapture.Truncated(),
	}
}

func commandStartFailure(started time.Time, err error) CommandResult {
	return CommandResult{
		ExitCode:   -1,
		Stderr:     err.Error(),
		DurationMS: time.Since(started).Milliseconds(),
	}
}

func commandEnvironment(environment []string) []string {
	blocked := map[string]struct{}{
		envAPIKey:   {},
		envEndpoint: {},
		envBaseURL:  {},
		envModel:    {},
	}
	cleaned := make([]string, 0, len(environment))
	for _, item := range environment {
		name, _, found := strings.Cut(item, "=")
		if _, shouldBlock := blocked[name]; found && shouldBlock {
			continue
		}
		cleaned = append(cleaned, item)
	}
	return cleaned
}

type cappedBuffer struct {
	limit     int
	headLimit int
	data      []byte
	truncated bool
}

func newCappedBuffer(limit int) *cappedBuffer {
	return &cappedBuffer{limit: limit, headLimit: limit / 2}
}

func (b *cappedBuffer) Write(p []byte) (int, error) {
	originalLength := len(p)
	if b.limit <= 0 {
		b.truncated = b.truncated || len(p) > 0
		return originalLength, nil
	}

	if !b.truncated && len(b.data)+len(p) <= b.limit {
		b.data = append(b.data, p...)
		return originalLength, nil
	}

	tailLimit := b.limit - b.headLimit
	if !b.truncated {
		combined := make([]byte, 0, len(b.data)+len(p))
		combined = append(combined, b.data...)
		combined = append(combined, p...)
		head := append([]byte(nil), combined[:b.headLimit]...)
		tailStart := len(combined) - tailLimit
		if tailStart < b.headLimit {
			tailStart = b.headLimit
		}
		b.data = append(head, combined[tailStart:]...)
		b.truncated = true
		return originalLength, nil
	}

	head := append([]byte(nil), b.data[:b.headLimit]...)
	currentTail := b.data[b.headLimit:]
	newTail := make([]byte, 0, len(currentTail)+len(p))
	newTail = append(newTail, currentTail...)
	newTail = append(newTail, p...)
	if len(newTail) > tailLimit {
		newTail = newTail[len(newTail)-tailLimit:]
	}
	b.data = append(head, newTail...)
	return originalLength, nil
}

func (b *cappedBuffer) String() string {
	if !b.truncated {
		return string(b.data)
	}
	return string(b.data[:b.headLimit]) + "\n... output truncated ...\n" + string(b.data[b.headLimit:])
}

func (b *cappedBuffer) Truncated() bool { return b.truncated }

var _ io.Writer = (*cappedBuffer)(nil)
