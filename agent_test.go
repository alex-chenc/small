package main

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
	"time"
)

func TestAgentRunsCommandThroughChatCompletions(t *testing.T) {
	var (
		mu       sync.Mutex
		requests []chatRequest
	)

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			t.Errorf("method = %s, want POST", r.Method)
		}
		if r.Header.Get("Authorization") != "Bearer test-key" {
			t.Errorf("authorization = %q", r.Header.Get("Authorization"))
		}

		body, err := io.ReadAll(r.Body)
		if err != nil {
			t.Errorf("read request: %v", err)
			return
		}
		var request chatRequest
		if err := json.Unmarshal(body, &request); err != nil {
			t.Errorf("decode request: %v", err)
			return
		}
		mu.Lock()
		requests = append(requests, request)
		requestNumber := len(requests)
		mu.Unlock()

		switch requestNumber {
		case 1:
			w.Header().Set("Content-Type", "text/event-stream")
			writeSSE(t, w, map[string]any{
				"choices": []any{map[string]any{"delta": map[string]any{
					"role": "assistant", "content": "\n\nPlan: run the test command.\n\n",
				}}},
			})
			writeSSE(t, w, map[string]any{
				"choices": []any{map[string]any{"delta": map[string]any{
					"tool_calls": []any{map[string]any{
						"index": 0, "id": "call_1", "type": "function",
						"function": map[string]any{"name": "exec_command", "arguments": `{"cmd":"printf `},
					}},
				}}},
			})
			writeSSE(t, w, map[string]any{
				"choices": []any{map[string]any{"delta": map[string]any{
					"tool_calls": []any{map[string]any{
						"index":    0,
						"function": map[string]any{"arguments": `integration-ok","timeout_ms":5000}`},
					}},
				}}},
			})
			writeSSEDone(w)
			return
		case 2:
			w.Header().Set("Content-Type", "application/json")
			_, _ = io.WriteString(w, `{"choices":[{"message":{"role":"assistant","content":"Done: the command completed successfully.","tool_calls":[{"id":"finish_1","type":"function","function":{"name":"finish_task","arguments":"{\"summary\":\"Done: the command completed successfully.\"}"}}]}}]}`)
			return
		default:
			t.Errorf("unexpected request number %d", requestNumber)
			http.Error(w, "unexpected request", http.StatusInternalServerError)
		}
	}))
	defer server.Close()

	var stdout, stderr bytes.Buffer
	client := &OpenAIClient{
		apiKey:       "test-key",
		endpoint:     server.URL,
		model:        "test-model",
		instructions: "test instructions",
		httpClient:   server.Client(),
	}
	agent := Agent{
		client:   client,
		executor: &CommandExecutor{cwd: t.TempDir()},
		stdout:   &stdout,
		stderr:   &stderr,
	}
	if err := agent.Run(context.Background(), "run the integration test"); err != nil {
		t.Fatalf("Agent.Run() error = %v", err)
	}

	output := stdout.String()
	for _, expected := range []string{
		"◆ Starting task: run the integration test",
		"● Plan: run the test command.",
		"▶ $ printf integration-ok",
		"│ integration-ok",
		"└ Exit code 0",
		"● Done: the command completed successfully.",
		"◆ Task complete",
	} {
		if !strings.Contains(output, expected) {
			t.Errorf("stdout does not contain %q:\n%s", expected, output)
		}
	}
	if stderr.Len() != 0 {
		t.Errorf("stderr = %q", stderr.String())
	}
	if strings.Contains(output, "\n\n") {
		t.Errorf("stdout contains an oversized blank interval:\n%q", output)
	}

	mu.Lock()
	defer mu.Unlock()
	if len(requests) != 2 {
		t.Fatalf("request count = %d, want 2", len(requests))
	}
	if !requests[0].Stream || requests[0].ToolChoice != "auto" {
		t.Errorf("unexpected request settings: %+v", requests[0])
	}
	if len(requests[0].Tools) != 3 ||
		requests[0].Tools[0].Function.Name != "exec_command" ||
		requests[0].Tools[1].Function.Name != "request_user_input" ||
		requests[0].Tools[2].Function.Name != "finish_task" {
		t.Errorf("unexpected tools: %+v", requests[0].Tools)
	}
	if len(requests[0].Messages) != 2 || requests[0].Messages[0].Role != "system" || requests[0].Messages[1].Role != "user" {
		t.Errorf("unexpected initial messages: %+v", requests[0].Messages)
	}

	var foundOutput bool
	for _, message := range requests[1].Messages {
		if message.Role != "tool" {
			continue
		}
		foundOutput = true
		if message.ToolCallID != "call_1" {
			t.Errorf("tool_call_id = %q", message.ToolCallID)
		}
		var result CommandResult
		if err := json.Unmarshal([]byte(message.Content), &result); err != nil {
			t.Fatalf("decode tool output: %v", err)
		}
		if result.ExitCode != 0 || result.Stdout != "integration-ok" {
			t.Fatalf("tool result = %+v", result)
		}
	}
	if !foundOutput {
		t.Fatal("second request did not contain a tool message")
	}
}

func TestAgentRequestsUserInputAndContinues(t *testing.T) {
	var requests int
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		requests++
		var request chatRequest
		if err := json.NewDecoder(r.Body).Decode(&request); err != nil {
			t.Errorf("decode request: %v", err)
			return
		}

		w.Header().Set("Content-Type", "application/json")
		switch requests {
		case 1:
			_, _ = io.WriteString(w, `{"choices":[{"message":{"role":"assistant","content":"I need confirmation.","tool_calls":[{"id":"input_1","type":"function","function":{"name":"request_user_input","arguments":"{\"question\":\"Proceed with the test?\"}"}}]}}]}`)
			return
		case 2:
			var foundAnswer bool
			for _, message := range request.Messages {
				if message.Role != "tool" || message.ToolCallID != "input_1" {
					continue
				}
				var result requestUserInputResult
				if err := json.Unmarshal([]byte(message.Content), &result); err != nil {
					t.Errorf("decode input result: %v", err)
					continue
				}
				foundAnswer = result.Answer == "yes" && result.Error == ""
			}
			if !foundAnswer {
				t.Errorf("request did not contain the expected user answer: %+v", request.Messages)
			}
			_, _ = io.WriteString(w, `{"choices":[{"message":{"role":"assistant","content":"Confirmed.","tool_calls":[{"id":"finish_1","type":"function","function":{"name":"finish_task","arguments":"{\"summary\":\"Confirmed.\"}"}}]}}]}`)
			return
		default:
			t.Errorf("unexpected request number %d", requests)
			http.Error(w, "unexpected request", http.StatusInternalServerError)
		}
	}))
	defer server.Close()

	var stdout, stderr bytes.Buffer
	agent := Agent{
		client: &OpenAIClient{
			apiKey:       "test-key",
			endpoint:     server.URL,
			model:        "test-model",
			instructions: "test instructions",
			httpClient:   server.Client(),
		},
		executor: &CommandExecutor{cwd: t.TempDir()},
		stdin:    strings.NewReader("yes\n"),
		stdout:   &stdout,
		stderr:   &stderr,
	}
	if err := agent.Run(context.Background(), "confirm the test"); err != nil {
		t.Fatalf("Agent.Run() error = %v", err)
	}
	if requests != 2 {
		t.Fatalf("request count = %d, want 2", requests)
	}
	for _, expected := range []string{
		"● I need confirmation.",
		"? Proceed with the test?",
		"> \n└ Input received",
		"● Confirmed.",
		"◆ Task complete",
	} {
		if !strings.Contains(stdout.String(), expected) {
			t.Errorf("stdout does not contain %q:\n%s", expected, stdout.String())
		}
	}
	if stderr.Len() != 0 {
		t.Errorf("stderr = %q", stderr.String())
	}
}

func TestAgentUsesModelDecisionForPlainTextInputRequest(t *testing.T) {
	var requests int
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		requests++
		var request chatRequest
		if err := json.NewDecoder(r.Body).Decode(&request); err != nil {
			t.Errorf("decode request: %v", err)
			return
		}

		w.Header().Set("Content-Type", "application/json")
		switch requests {
		case 1:
			_, _ = io.WriteString(w, `{"choices":[{"message":{"role":"assistant","content":"Available options:\nA. Do nothing\nB. Clear cache\nTell me which option to run."}}]}`)
		case 2:
			if request.ToolChoice != "none" || request.Stream || len(request.Tools) != 0 {
				t.Errorf("unexpected control decision request: %+v", request)
			}
			_, _ = io.WriteString(w, `{"choices":[{"message":{"role":"assistant","content":"{\"action\":\"request_user_input\",\"question\":\"Which option should I run?\"}"}}]}`)
		case 3:
			if len(request.Messages) == 0 {
				t.Error("continuation request did not contain messages")
			} else {
				last := request.Messages[len(request.Messages)-1]
				if last.Role != "user" || last.Content != "B" {
					t.Errorf("last continuation message = %+v, want user answer B", last)
				}
			}
			_, _ = io.WriteString(w, `{"choices":[{"message":{"role":"assistant","content":"Option B selected.","tool_calls":[{"id":"finish_1","type":"function","function":{"name":"finish_task","arguments":"{\"summary\":\"Option B selected.\"}"}}]}}]}`)
		default:
			t.Errorf("unexpected request number %d", requests)
			http.Error(w, "unexpected request", http.StatusInternalServerError)
		}
	}))
	defer server.Close()

	var stdout, stderr bytes.Buffer
	agent := Agent{
		client: &OpenAIClient{
			apiKey:       "test-key",
			endpoint:     server.URL,
			model:        "test-model",
			instructions: "test instructions",
			httpClient:   server.Client(),
		},
		executor: &CommandExecutor{cwd: t.TempDir()},
		stdin:    strings.NewReader("B\n"),
		stdout:   &stdout,
		stderr:   &stderr,
	}
	if err := agent.Run(context.Background(), "choose a test option"); err != nil {
		t.Fatalf("Agent.Run() error = %v", err)
	}
	if requests != 3 {
		t.Fatalf("request count = %d, want 3", requests)
	}
	for _, expected := range []string{
		"● Available options:",
		"◆ Resolving next action...",
		"? Which option should I run?",
		"> \n└ Input received",
		"● Option B selected.",
		"◆ Task complete",
	} {
		if !strings.Contains(stdout.String(), expected) {
			t.Errorf("stdout does not contain %q:\n%s", expected, stdout.String())
		}
	}
	if stderr.Len() != 0 {
		t.Errorf("stderr = %q", stderr.String())
	}
}

func TestAgentUsesFallbackDecisionForPlainTextFinalResponse(t *testing.T) {
	var requests []chatRequest
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var request chatRequest
		if err := json.NewDecoder(r.Body).Decode(&request); err != nil {
			t.Errorf("decode request: %v", err)
			return
		}
		requests = append(requests, request)
		w.Header().Set("Content-Type", "application/json")
		switch len(requests) {
		case 1:
			_, _ = io.WriteString(w, `{"choices":[{"message":{"role":"assistant","content":"The task is already complete."}}]}`)
		case 2:
			_, _ = io.WriteString(w, `{"choices":[{"message":{"role":"assistant","content":"{\"action\":\"finish\"}"}}]}`)
		default:
			t.Errorf("unexpected request number %d", len(requests))
			http.Error(w, "unexpected request", http.StatusInternalServerError)
		}
	}))
	defer server.Close()

	var stdout, stderr bytes.Buffer
	agent := Agent{
		client: &OpenAIClient{
			apiKey:       "test-key",
			endpoint:     server.URL,
			model:        "test-model",
			instructions: "test instructions",
			httpClient:   server.Client(),
		},
		executor: &CommandExecutor{cwd: t.TempDir()},
		stdout:   &stdout,
		stderr:   &stderr,
	}
	if err := agent.Run(context.Background(), "report completion"); err != nil {
		t.Fatalf("Agent.Run() error = %v", err)
	}
	if len(requests) != 2 {
		t.Fatalf("request count = %d, want 2", len(requests))
	}
	if requests[1].ToolChoice != "none" || requests[1].Stream || len(requests[1].Tools) != 0 {
		t.Errorf("unexpected fallback decision request: %+v", requests[1])
	}
	for _, expected := range []string{
		"● The task is already complete.",
		"◆ Resolving next action...",
		"◆ Task complete",
	} {
		if !strings.Contains(stdout.String(), expected) {
			t.Errorf("stdout does not contain %q:\n%s", expected, stdout.String())
		}
	}
	if stderr.Len() != 0 {
		t.Errorf("stderr = %q", stderr.String())
	}
}

func TestParseControlDecision(t *testing.T) {
	tests := []struct {
		name          string
		text          string
		want          controlDecision
		wantErrorPart string
	}{
		{name: "finish", text: `{"action":"finish"}`, want: controlDecision{Action: "finish"}},
		{
			name: "input with markdown wrapper",
			text: "```json\n{\"action\":\"request_user_input\",\"question\":\"  Which   option?  \"}\n```",
			want: controlDecision{Action: "request_user_input", Question: "Which option?"},
		},
		{name: "missing question", text: `{"action":"request_user_input"}`, wantErrorPart: "without a question"},
		{name: "unknown action", text: `{"action":"continue"}`, wantErrorPart: "unsupported"},
		{name: "invalid response", text: "finish", wantErrorPart: "JSON object"},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			got, err := parseControlDecision(test.text)
			if test.wantErrorPart != "" {
				if err == nil || !strings.Contains(err.Error(), test.wantErrorPart) {
					t.Fatalf("parseControlDecision() error = %v, want containing %q", err, test.wantErrorPart)
				}
				return
			}
			if err != nil {
				t.Fatalf("parseControlDecision() error = %v", err)
			}
			if got != test.want {
				t.Fatalf("parseControlDecision() = %+v, want %+v", got, test.want)
			}
		})
	}
}

func TestPrefixedWriterMarksEveryOutputLine(t *testing.T) {
	var stdout, stderr bytes.Buffer
	terminalStdout, _ := newTerminalOutputs(&stdout, &stderr)
	writer := &prefixedWriter{output: terminalStdout, prefix: "│ "}

	_, _ = io.WriteString(writer, "first line\n\n")
	_, _ = io.WriteString(writer, "second line")
	terminalStdout.ensureLineStart()

	want := "│ first line\n│ \n│ second line\n"
	if stdout.String() != want {
		t.Fatalf("prefixed output = %q, want %q", stdout.String(), want)
	}
}

func TestOpenAIClientAcceptsNonStreamingChatResponse(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `{"choices":[{"message":{"role":"assistant","content":"Hello"}}]}`)
	}))
	defer server.Close()

	client := OpenAIClient{
		apiKey: "key", endpoint: server.URL, model: "model", httpClient: server.Client(),
	}
	response, err := client.createResponse(context.Background(), []chatMessage{{Role: "user", Content: "Hello"}}, nil, nil)
	if err != nil {
		t.Fatalf("createResponse() error = %v", err)
	}
	if response.Message.Content != "Hello" || response.TextStreamed {
		t.Fatalf("response = %+v", response)
	}
}

func TestOpenAIClientRetriesRateLimitAndReplaysRequest(t *testing.T) {
	var requests int
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		requests++
		if requests <= 2 {
			w.Header().Set("Retry-After", "0")
			w.WriteHeader(http.StatusTooManyRequests)
			_, _ = io.WriteString(w, `{"error":{"message":"rate limited","type":"rate_limit_error","code":"429"}}`)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `{"choices":[{"message":{"role":"assistant","content":"Retry succeeded"}}]}`)
	}))
	defer server.Close()

	client := OpenAIClient{
		apiKey: "key", endpoint: server.URL, model: "model", httpClient: server.Client(),
	}
	var retries []apiRetryEvent
	response, err := client.createResponse(
		context.Background(),
		[]chatMessage{{Role: "user", Content: "Hello"}},
		nil,
		func(event apiRetryEvent) { retries = append(retries, event) },
	)
	if err != nil {
		t.Fatalf("createResponse() error = %v", err)
	}
	if response.Message.Content != "Retry succeeded" {
		t.Fatalf("response content = %q", response.Message.Content)
	}
	if requests != 3 || len(retries) != 2 {
		t.Fatalf("requests = %d, retries = %d; want 3 and 2", requests, len(retries))
	}
	for index, event := range retries {
		if event.StatusCode != http.StatusTooManyRequests || event.Attempt != index+1 || event.Delay != 0 {
			t.Errorf("retry[%d] = %+v", index, event)
		}
	}
}

func TestOpenAIClientStopsAfterRateLimitRetries(t *testing.T) {
	var requests int
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		requests++
		w.Header().Set("Retry-After", "0")
		w.WriteHeader(http.StatusTooManyRequests)
		_, _ = io.WriteString(w, `{"error":{"message":"still limited"}}`)
	}))
	defer server.Close()

	client := OpenAIClient{
		apiKey: "key", endpoint: server.URL, model: "model", httpClient: server.Client(),
	}
	_, err := client.createResponse(context.Background(), []chatMessage{{Role: "user", Content: "Hello"}}, nil, nil)
	if err == nil || !strings.Contains(err.Error(), "still failing after 4 retries") {
		t.Fatalf("createResponse() error = %v", err)
	}
	if requests != maxAPIRetries+1 {
		t.Fatalf("requests = %d, want %d", requests, maxAPIRetries+1)
	}
}

func TestRetryDelay(t *testing.T) {
	now := time.Date(2026, time.August, 14, 10, 0, 0, 0, time.UTC)
	tests := []struct {
		header     string
		retryIndex int
		want       time.Duration
	}{
		{header: "0.25", want: 250 * time.Millisecond},
		{header: now.Add(3 * time.Second).Format(http.TimeFormat), want: 3 * time.Second},
		{retryIndex: 0, want: time.Second},
		{retryIndex: 3, want: 8 * time.Second},
	}
	for _, test := range tests {
		if got := retryDelay(test.header, test.retryIndex, now); got != test.want {
			t.Errorf("retryDelay(%q, %d) = %s, want %s", test.header, test.retryIndex, got, test.want)
		}
	}
}

func writeSSE(t *testing.T, w http.ResponseWriter, value any) {
	t.Helper()
	encoded, err := json.Marshal(value)
	if err != nil {
		t.Fatalf("marshal SSE event: %v", err)
	}
	_, _ = fmt.Fprintf(w, "data: %s\n\n", encoded)
	flush(w)
}

func writeSSEDone(w http.ResponseWriter) {
	_, _ = io.WriteString(w, "data: [DONE]\n\n")
	flush(w)
}

func flush(w http.ResponseWriter) {
	if flusher, ok := w.(http.Flusher); ok {
		flusher.Flush()
	}
}
