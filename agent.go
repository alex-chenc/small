package main

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"strings"
	"sync"
	"time"
	"unicode"
)

const maxAgentTurns = 30

type Agent struct {
	client   *OpenAIClient
	executor *CommandExecutor
	stdin    io.Reader
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

type requestUserInputArguments struct {
	Question string `json:"question"`
}

type requestUserInputResult struct {
	Answer string `json:"answer"`
	Error  string `json:"error,omitempty"`
}

type toolErrorResult struct {
	Error string `json:"error"`
}

type inputReadResult struct {
	line string
	err  error
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

func (o *terminalOutput) markLineStart() {
	o.state.mu.Lock()
	defer o.state.mu.Unlock()
	o.state.atLineStart = true
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
	input := a.stdin
	if input == nil {
		input = strings.NewReader("")
	}
	inputReader := bufio.NewReader(input)
	interactiveInput := isTerminalStream(input) && isTerminalStream(a.stdout)

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
			inputRequired := responseRequestsInput(response.Message.Content)
			if inputRequired || interactiveInput {
				stdout.ensureLineStart()
				if inputRequired {
					fmt.Fprintln(stdout, "? Enter your response to continue:")
				} else {
					fmt.Fprintln(stdout, "? Follow-up (press Enter to finish):")
				}
				fmt.Fprint(stdout, "> ")
				answer, err := readUserInput(ctx, inputReader)
				finishInputPrompt(stdout, interactiveInput)
				if err != nil {
					if ctx.Err() != nil {
						return ctx.Err()
					}
					if inputRequired {
						fmt.Fprintf(stdout, "└ Input unavailable: %s\n", err)
					}
					return nil
				}
				answer = strings.TrimSpace(answer)
				if answer == "" && !inputRequired {
					fmt.Fprintln(stdout, "└ Finished")
					return nil
				}
				fmt.Fprintln(stdout, "└ Input received")
				messages = append(messages, chatMessage{Role: "user", Content: answer})
				continue
			}
			stdout.ensureLineStart()
			return nil
		}

		hasInputRequest := false
		for _, toolCall := range response.Message.ToolCalls {
			if toolCall.Function.Name == "request_user_input" {
				hasInputRequest = true
				break
			}
		}
		inputHandled := false
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

			var result any
			if hasInputRequest && call.Name != "request_user_input" {
				result = toolErrorResult{Error: "tool call was not executed because user input was requested in the same response; consider the user's answer before calling another tool"}
			} else if call.Name == "request_user_input" && inputHandled {
				result = toolErrorResult{Error: "only one request_user_input call is allowed per response"}
			} else {
				if call.Name == "request_user_input" {
					inputHandled = true
				}
				result, err = a.executeCall(ctx, call, inputReader, interactiveInput, stdout, stderr)
				if err != nil {
					return err
				}
			}
			resultJSON, err := json.Marshal(result)
			if err != nil {
				return fmt.Errorf("encode tool result: %w", err)
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

func responseRequestsInput(text string) bool {
	if containsChoiceList(text) {
		return true
	}

	runes := []rune(strings.TrimSpace(text))
	const tailLength = 512
	if len(runes) > tailLength {
		runes = runes[len(runes)-tailLength:]
	}

	for index, current := range runes {
		if current == '\uFF1F' {
			return true
		}
		if current != '?' {
			continue
		}
		if index == len(runes)-1 {
			return true
		}
		next := runes[index+1]
		if unicode.IsSpace(next) || strings.ContainsRune("\"'`*_)]}", next) {
			return true
		}
	}
	return false
}

func containsChoiceList(text string) bool {
	seen := make(map[string]struct{})
	for _, line := range strings.Split(text, "\n") {
		line = strings.TrimLeft(strings.TrimSpace(line), "-+*`> \t")
		runes := []rune(line)
		if len(runes) < 2 {
			continue
		}

		var key string
		switch {
		case runes[0] >= 'A' && runes[0] <= 'Z' && isChoiceSeparator(runes[1]):
			key = string(runes[0])
		case runes[0] >= '0' && runes[0] <= '9':
			end := 1
			for end < len(runes) && runes[end] >= '0' && runes[end] <= '9' {
				end++
			}
			if end < len(runes) && isChoiceSeparator(runes[end]) {
				key = string(runes[:end])
			}
		}
		if key == "" {
			continue
		}
		seen[key] = struct{}{}
		if len(seen) >= 2 {
			return true
		}
	}
	return false
}

func isChoiceSeparator(value rune) bool {
	return value == '.' || value == ')' || value == ':'
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

func (a *Agent) executeCall(
	ctx context.Context,
	call functionCall,
	input *bufio.Reader,
	interactiveInput bool,
	stdout, stderr *terminalOutput,
) (any, error) {
	switch call.Name {
	case "exec_command":
		var args execCommandArguments
		if err := json.Unmarshal([]byte(call.Arguments), &args); err != nil {
			return CommandResult{ExitCode: -1, Stderr: "exec_command arguments are not valid JSON: " + err.Error()}, nil
		}
		args.Command = strings.TrimSpace(args.Command)
		if args.Command == "" {
			return CommandResult{ExitCode: -1, Stderr: "exec_command.cmd must not be empty"}, nil
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
		return result, nil

	case "request_user_input":
		var args requestUserInputArguments
		if err := json.Unmarshal([]byte(call.Arguments), &args); err != nil {
			return requestUserInputResult{Error: "request_user_input arguments are not valid JSON: " + err.Error()}, nil
		}
		question := strings.Join(strings.Fields(args.Question), " ")
		if question == "" {
			return requestUserInputResult{Error: "request_user_input.question must not be empty"}, nil
		}

		stdout.ensureLineStart()
		fmt.Fprintf(stdout, "? %s\n", question)
		fmt.Fprint(stdout, "> ")
		answer, err := readUserInput(ctx, input)
		finishInputPrompt(stdout, interactiveInput)
		if err != nil {
			if ctx.Err() != nil {
				return nil, ctx.Err()
			}
			fmt.Fprintf(stdout, "└ Input unavailable: %s\n", err)
			return requestUserInputResult{Error: err.Error()}, nil
		}
		fmt.Fprintln(stdout, "└ Input received")
		return requestUserInputResult{Answer: strings.TrimSpace(answer)}, nil

	default:
		return CommandResult{ExitCode: -1, Stderr: "unsupported tool: " + call.Name}, nil
	}
}

func isTerminalStream(stream any) bool {
	file, ok := stream.(*os.File)
	if !ok {
		return false
	}
	info, err := file.Stat()
	return err == nil && info.Mode()&os.ModeCharDevice != 0
}

func finishInputPrompt(output *terminalOutput, interactive bool) {
	if interactive {
		output.markLineStart()
		return
	}
	output.ensureLineStart()
}

func readUserInput(ctx context.Context, input *bufio.Reader) (string, error) {
	resultCh := make(chan inputReadResult, 1)
	go func() {
		line, err := input.ReadString('\n')
		resultCh <- inputReadResult{line: line, err: err}
	}()

	select {
	case <-ctx.Done():
		return "", ctx.Err()
	case result := <-resultCh:
		if result.err == nil {
			return result.line, nil
		}
		if errors.Is(result.err, io.EOF) {
			if result.line != "" {
				return result.line, nil
			}
			return "", errors.New("stdin closed before user input was received")
		}
		return "", fmt.Errorf("read user input: %w", result.err)
	}
}
