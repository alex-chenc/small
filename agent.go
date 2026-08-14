package main

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"strings"
	"sync"
	"time"
)

const maxAgentTurns = 30

type Agent struct {
	client   *OpenAIClient
	executor *CommandExecutor
	stdout   io.Writer
	stderr   io.Writer
}

type functionCall struct {
	CallID    string
	Name      string
	Arguments string
}

type execCommandArguments struct {
	Command   string `json:"cmd"`
	TimeoutMS int64  `json:"timeout_ms"`
}

type outputState struct {
	mu          sync.Mutex
	atLineStart bool
}

type terminalOutput struct {
	writer io.Writer
	state  *outputState
}

type prefixedWriter struct {
	output *terminalOutput
	prefix string
}

func newTerminalOutputs(stdout, stderr io.Writer) (*terminalOutput, *terminalOutput) {
	state := &outputState{atLineStart: true}
	return &terminalOutput{writer: stdout, state: state}, &terminalOutput{writer: stderr, state: state}
}

func (o *terminalOutput) Write(p []byte) (int, error) {
	o.state.mu.Lock()
	defer o.state.mu.Unlock()

	n, err := o.writer.Write(p)
	if n > 0 {
		o.state.atLineStart = p[n-1] == '\n' || p[n-1] == '\r'
	}
	return n, err
}

func (o *terminalOutput) writePrefixed(p []byte, prefix string) (int, error) {
	o.state.mu.Lock()
	defer o.state.mu.Unlock()

	consumed := 0
	for consumed < len(p) {
		if o.state.atLineStart {
			if _, err := io.WriteString(o.writer, prefix); err != nil {
				return consumed, err
			}
			o.state.atLineStart = false
		}

		remaining := p[consumed:]
		segmentLength := len(remaining)
		if lineBreak := bytes.IndexAny(remaining, "\r\n"); lineBreak >= 0 {
			segmentLength = lineBreak + 1
			if remaining[lineBreak] == '\r' && segmentLength < len(remaining) && remaining[segmentLength] == '\n' {
				segmentLength++
			}
		}

		n, err := o.writer.Write(remaining[:segmentLength])
		consumed += n
		if n > 0 {
			last := remaining[n-1]
			o.state.atLineStart = last == '\n' || last == '\r'
		}
		if err != nil {
			return consumed, err
		}
		if n != segmentLength {
			return consumed, io.ErrShortWrite
		}
	}
	return consumed, nil
}

func (w *prefixedWriter) Write(p []byte) (int, error) {
	return w.output.writePrefixed(p, w.prefix)
}

func (o *terminalOutput) ensureLineStart() {
	o.state.mu.Lock()
	defer o.state.mu.Unlock()
	if o.state.atLineStart {
		return
	}
	if _, err := io.WriteString(o.writer, "\n"); err == nil {
		o.state.atLineStart = true
	}
}

type compactModelOutput struct {
	output       io.Writer
	started      bool
	pendingBreak bool
}

func (o *compactModelOutput) WriteText(text string) {
	for text != "" {
		newline := strings.IndexAny(text, "\r\n")
		if newline < 0 {
			o.writeContent(text)
			return
		}
		if newline > 0 {
			o.writeContent(text[:newline])
		}
		o.pendingBreak = o.started
		text = strings.TrimLeft(text[newline:], "\r\n")
	}
}

func (o *compactModelOutput) writeContent(text string) {
	if text == "" {
		return
	}
	if o.pendingBreak {
		_, _ = io.WriteString(o.output, "\n")
		o.pendingBreak = false
	}
	_, _ = io.WriteString(o.output, text)
	o.started = true
}

func (a *Agent) Run(ctx context.Context, task string) error {
	stdout, stderr := newTerminalOutputs(a.stdout, a.stderr)
	fmt.Fprintf(stdout, "◆ Starting task: %s\n", task)

	messages := []chatMessage{
		{Role: "system", Content: a.client.instructions},
		{Role: "user", Content: task},
	}
	executed := make(map[string]chatMessage)

	for turn := 0; turn < maxAgentTurns; turn++ {
		if err := ctx.Err(); err != nil {
			return err
		}

		modelOutput := &compactModelOutput{
			output: &prefixedWriter{output: stdout, prefix: "● "},
		}
		response, err := a.client.createResponse(
			ctx,
			messages,
			func(delta string) {
				modelOutput.WriteText(delta)
			},
			func(event apiRetryEvent) {
				reason := "network error"
				if event.StatusCode != 0 {
					reason = fmt.Sprintf("HTTP %d", event.StatusCode)
				}
				stdout.ensureLineStart()
				fmt.Fprintf(
					stdout,
					"◆ OpenAPI temporarily unavailable: %s; retrying in %s (%d/%d)\n",
					reason,
					formatRetryDelay(event.Delay),
					event.Attempt,
					event.MaxRetries,
				)
			},
		)
		if err != nil {
			return err
		}
		if !response.TextStreamed {
			modelOutput.WriteText(response.Message.Content)
		}

		messages = append(messages, response.Message)
		if len(response.Message.ToolCalls) == 0 {
			stdout.ensureLineStart()
			return nil
		}

		for _, toolCall := range response.Message.ToolCalls {
			if err := ctx.Err(); err != nil {
				return err
			}
			if toolCall.ID == "" {
				return errors.New("tool call is missing an id")
			}
			call := functionCall{
				CallID:    toolCall.ID,
				Name:      toolCall.Function.Name,
				Arguments: toolCall.Function.Arguments,
			}
			if old, ok := executed[call.CallID]; ok {
				messages = append(messages, old)
				continue
			}

			result := a.executeCall(ctx, call, stdout, stderr)
			resultJSON, err := json.Marshal(result)
			if err != nil {
				return fmt.Errorf("encode command result: %w", err)
			}
			outputMessage := chatMessage{
				Role:       "tool",
				ToolCallID: call.CallID,
				Content:    string(resultJSON),
			}
			executed[call.CallID] = outputMessage
			messages = append(messages, outputMessage)
		}
	}

	return fmt.Errorf("agent exceeded the maximum turn limit of %d", maxAgentTurns)
}

func formatRetryDelay(delay time.Duration) string {
	if delay < time.Second {
		return fmt.Sprintf("%dms", delay.Milliseconds())
	}
	if delay%time.Second == 0 {
		return fmt.Sprintf("%ds", int(delay/time.Second))
	}
	return delay.Round(100 * time.Millisecond).String()
}

func (a *Agent) executeCall(ctx context.Context, call functionCall, stdout, stderr *terminalOutput) CommandResult {
	if call.Name != "exec_command" {
		return CommandResult{ExitCode: -1, Stderr: "unsupported tool: " + call.Name}
	}

	var args execCommandArguments
	if err := json.Unmarshal([]byte(call.Arguments), &args); err != nil {
		return CommandResult{ExitCode: -1, Stderr: "exec_command arguments are not valid JSON: " + err.Error()}
	}
	args.Command = strings.TrimSpace(args.Command)
	if args.Command == "" {
		return CommandResult{ExitCode: -1, Stderr: "exec_command.cmd must not be empty"}
	}

	stdout.ensureLineStart()
	fmt.Fprintf(stdout, "▶ $ %s\n", args.Command)
	toolStdout := &prefixedWriter{output: stdout, prefix: "│ "}
	toolStderr := &prefixedWriter{output: stderr, prefix: "│ "}
	result := a.executor.Execute(ctx, args.Command, time.Duration(args.TimeoutMS)*time.Millisecond, toolStdout, toolStderr)
	stdout.ensureLineStart()
	if result.TimedOut {
		fmt.Fprintf(stdout, "└ Command timed out, exit code %d, duration %dms\n", result.ExitCode, result.DurationMS)
	} else {
		fmt.Fprintf(stdout, "└ Exit code %d, duration %dms\n", result.ExitCode, result.DurationMS)
	}
	return result
}
