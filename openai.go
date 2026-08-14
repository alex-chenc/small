package main

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"sort"
	"strconv"
	"strings"
	"time"
)

const (
	maxAPIErrorBytes = 1 << 20
	maxAPIRetries    = 4
	maxRetryDelay    = 30 * time.Second
)

type OpenAIClient struct {
	apiKey       string
	endpoint     string
	model        string
	instructions string
	httpClient   *http.Client
}

type chatRequest struct {
	Model      string               `json:"model"`
	Messages   []chatMessage        `json:"messages"`
	Tools      []chatToolDefinition `json:"tools"`
	ToolChoice string               `json:"tool_choice"`
	Stream     bool                 `json:"stream"`
}

type chatMessage struct {
	Role       string         `json:"role"`
	Content    string         `json:"content,omitempty"`
	ToolCalls  []chatToolCall `json:"tool_calls,omitempty"`
	ToolCallID string         `json:"tool_call_id,omitempty"`
}

type chatToolCall struct {
	ID       string           `json:"id"`
	Type     string           `json:"type"`
	Function chatFunctionCall `json:"function"`
}

type chatFunctionCall struct {
	Name      string `json:"name"`
	Arguments string `json:"arguments"`
}

type chatToolDefinition struct {
	Type     string                 `json:"type"`
	Function chatFunctionDefinition `json:"function"`
}

type chatFunctionDefinition struct {
	Name        string         `json:"name"`
	Description string         `json:"description"`
	Parameters  map[string]any `json:"parameters"`
}

type chatResponse struct {
	Message      chatMessage
	TextStreamed bool
}

type apiRetryEvent struct {
	StatusCode int
	Attempt    int
	MaxRetries int
	Delay      time.Duration
	Err        error
}

type apiError struct {
	Code    any    `json:"code"`
	Message string `json:"message"`
}

func (e *apiError) Error() string {
	if e == nil {
		return ""
	}
	if e.Code == nil || fmt.Sprint(e.Code) == "" {
		return e.Message
	}
	return fmt.Sprintf("%v: %s", e.Code, e.Message)
}

func execCommandTool() chatToolDefinition {
	return chatToolDefinition{
		Type: "function",
		Function: chatFunctionDefinition{
			Name:        "exec_command",
			Description: "Execute one command on the local Linux system using /bin/bash -lc and return its exit code, stdout, stderr, duration, and timeout status.",
			Parameters: map[string]any{
				"type":                 "object",
				"additionalProperties": false,
				"properties": map[string]any{
					"cmd": map[string]any{
						"type":        "string",
						"description": "The complete command to execute with /bin/bash -lc.",
					},
					"timeout_ms": map[string]any{
						"type":        "integer",
						"minimum":     1000,
						"maximum":     300000,
						"description": "Command timeout in milliseconds, between 1000 and 300000.",
					},
				},
				"required": []string{"cmd", "timeout_ms"},
			},
		},
	}
}

func requestUserInputTool() chatToolDefinition {
	return chatToolDefinition{
		Type: "function",
		Function: chatFunctionDefinition{
			Name:        "request_user_input",
			Description: "Ask the user one concise question and wait for one line of terminal input when required information or confirmation is missing. Returns JSON containing answer or error. Never request credentials or secrets.",
			Parameters: map[string]any{
				"type":                 "object",
				"additionalProperties": false,
				"properties": map[string]any{
					"question": map[string]any{
						"type":        "string",
						"description": "The single concise question to show to the user.",
					},
				},
				"required": []string{"question"},
			},
		},
	}
}

func finishTaskTool() chatToolDefinition {
	return chatToolDefinition{
		Type: "function",
		Function: chatFunctionDefinition{
			Name:        "finish_task",
			Description: "Finish the task when no command or user input is needed. The summary is shown as the final response. Call this tool alone.",
			Parameters: map[string]any{
				"type":                 "object",
				"additionalProperties": false,
				"properties": map[string]any{
					"summary": map[string]any{
						"type":        "string",
						"description": "A concise final summary of the completed task, verification result, and any unresolved problem.",
					},
				},
				"required": []string{"summary"},
			},
		},
	}
}

func agentTools() []chatToolDefinition {
	return []chatToolDefinition{execCommandTool(), requestUserInputTool(), finishTaskTool()}
}

func (c *OpenAIClient) createResponse(
	ctx context.Context,
	messages []chatMessage,
	onText func(string),
	onRetry func(apiRetryEvent),
) (chatResponse, error) {
	payload, err := json.Marshal(chatRequest{
		Model:      c.model,
		Messages:   messages,
		Tools:      agentTools(),
		ToolChoice: "required",
		Stream:     true,
	})
	if err != nil {
		return chatResponse{}, fmt.Errorf("encode API request: %w", err)
	}

	for attempt := 0; ; attempt++ {
		req, err := c.newRequest(ctx, payload)
		if err != nil {
			return chatResponse{}, err
		}

		resp, requestErr := c.httpClient.Do(req)
		if requestErr != nil {
			if ctx.Err() != nil {
				return chatResponse{}, ctx.Err()
			}
			if attempt >= maxAPIRetries {
				return chatResponse{}, fmt.Errorf("call OpenAI-compatible endpoint (failed after %d retries): %w", maxAPIRetries, requestErr)
			}
			delay := exponentialRetryDelay(attempt)
			notifyRetry(onRetry, apiRetryEvent{
				Attempt: attempt + 1, MaxRetries: maxAPIRetries, Delay: delay, Err: requestErr,
			})
			if err := waitForRetry(ctx, delay); err != nil {
				return chatResponse{}, err
			}
			continue
		}

		if resp.StatusCode < 200 || resp.StatusCode >= 300 {
			body, readErr := io.ReadAll(io.LimitReader(resp.Body, maxAPIErrorBytes))
			_ = resp.Body.Close()
			if readErr != nil {
				return chatResponse{}, fmt.Errorf("OpenAI-compatible endpoint returned HTTP %d and its error body could not be read: %w", resp.StatusCode, readErr)
			}
			bodyText := strings.TrimSpace(string(body))
			if isRetryableStatus(resp.StatusCode) && attempt < maxAPIRetries {
				delay := retryDelay(resp.Header.Get("Retry-After"), attempt, time.Now())
				notifyRetry(onRetry, apiRetryEvent{
					StatusCode: resp.StatusCode,
					Attempt:    attempt + 1,
					MaxRetries: maxAPIRetries,
					Delay:      delay,
				})
				if err := waitForRetry(ctx, delay); err != nil {
					return chatResponse{}, err
				}
				continue
			}
			if isRetryableStatus(resp.StatusCode) {
				return chatResponse{}, fmt.Errorf("OpenAI-compatible endpoint returned HTTP %d (still failing after %d retries): %s", resp.StatusCode, maxAPIRetries, bodyText)
			}
			return chatResponse{}, fmt.Errorf("OpenAI-compatible endpoint returned HTTP %d: %s", resp.StatusCode, bodyText)
		}

		defer resp.Body.Close()
		if strings.Contains(strings.ToLower(resp.Header.Get("Content-Type")), "text/event-stream") {
			return decodeChatSSE(resp.Body, onText)
		}
		return decodeChatJSON(resp.Body)
	}
}

func (c *OpenAIClient) newRequest(ctx context.Context, payload []byte) (*http.Request, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, c.endpoint, bytes.NewReader(payload))
	if err != nil {
		return nil, fmt.Errorf("create API request: %w", err)
	}
	req.Header.Set("Authorization", "Bearer "+c.apiKey)
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Accept", "text/event-stream")
	return req, nil
}

func notifyRetry(callback func(apiRetryEvent), event apiRetryEvent) {
	if callback != nil {
		callback(event)
	}
}

func isRetryableStatus(status int) bool {
	return status == http.StatusRequestTimeout ||
		status == http.StatusConflict ||
		status == http.StatusTooManyRequests ||
		status >= http.StatusInternalServerError
}

func retryDelay(header string, retryIndex int, now time.Time) time.Duration {
	header = strings.TrimSpace(header)
	if header != "" {
		if seconds, err := strconv.ParseFloat(header, 64); err == nil && seconds >= 0 {
			return capRetryDelay(time.Duration(seconds * float64(time.Second)))
		}
		if retryAt, err := http.ParseTime(header); err == nil {
			return capRetryDelay(max(retryAt.Sub(now), 0))
		}
	}
	return exponentialRetryDelay(retryIndex)
}

func exponentialRetryDelay(retryIndex int) time.Duration {
	delay := time.Second
	for range retryIndex {
		if delay >= maxRetryDelay/2 {
			return maxRetryDelay
		}
		delay *= 2
	}
	return capRetryDelay(delay)
}

func capRetryDelay(delay time.Duration) time.Duration {
	if delay < 0 {
		return 0
	}
	if delay > maxRetryDelay {
		return maxRetryDelay
	}
	return delay
}

func waitForRetry(ctx context.Context, delay time.Duration) error {
	if delay <= 0 {
		return ctx.Err()
	}
	timer := time.NewTimer(delay)
	defer timer.Stop()
	select {
	case <-timer.C:
		return nil
	case <-ctx.Done():
		return ctx.Err()
	}
}

type chatCompletion struct {
	Choices []struct {
		Message chatMessage `json:"message"`
	} `json:"choices"`
	Error *apiError `json:"error"`
}

func decodeChatJSON(r io.Reader) (chatResponse, error) {
	var completion chatCompletion
	if err := json.NewDecoder(r).Decode(&completion); err != nil {
		return chatResponse{}, fmt.Errorf("decode OpenAI-compatible response: %w", err)
	}
	if completion.Error != nil {
		return chatResponse{}, completion.Error
	}
	if len(completion.Choices) == 0 {
		return chatResponse{}, errors.New("OpenAI-compatible response contains no choices")
	}
	message := completion.Choices[0].Message
	if message.Role == "" {
		message.Role = "assistant"
	}
	return chatResponse{Message: message}, nil
}

type chatCompletionChunk struct {
	Choices []struct {
		Delta struct {
			Role      string `json:"role"`
			Content   string `json:"content"`
			ToolCalls []struct {
				Index    int    `json:"index"`
				ID       string `json:"id"`
				Type     string `json:"type"`
				Function struct {
					Name      string `json:"name"`
					Arguments string `json:"arguments"`
				} `json:"function"`
			} `json:"tool_calls"`
		} `json:"delta"`
	} `json:"choices"`
	Error *apiError `json:"error"`
}

func decodeChatSSE(r io.Reader, onText func(string)) (chatResponse, error) {
	var (
		content      strings.Builder
		partialCalls = make(map[int]*chatToolCall)
		streamedText bool
		sawEvent     bool
	)

	err := scanSSEData(r, func(data string) error {
		if data == "[DONE]" {
			return nil
		}
		var chunk chatCompletionChunk
		if err := json.Unmarshal([]byte(data), &chunk); err != nil {
			return fmt.Errorf("decode OpenAI-compatible SSE event: %w", err)
		}
		if chunk.Error != nil {
			return chunk.Error
		}
		sawEvent = true

		for _, choice := range chunk.Choices {
			if choice.Delta.Content != "" {
				streamedText = true
				content.WriteString(choice.Delta.Content)
				if onText != nil {
					onText(choice.Delta.Content)
				}
			}
			for _, delta := range choice.Delta.ToolCalls {
				call := partialCalls[delta.Index]
				if call == nil {
					call = &chatToolCall{Type: "function"}
					partialCalls[delta.Index] = call
				}
				if delta.ID != "" {
					call.ID = delta.ID
				}
				if delta.Type != "" {
					call.Type = delta.Type
				}
				call.Function.Name += delta.Function.Name
				call.Function.Arguments += delta.Function.Arguments
			}
		}
		return nil
	})
	if err != nil {
		return chatResponse{}, err
	}
	if !sawEvent {
		return chatResponse{}, errors.New("OpenAI-compatible stream ended before a valid event was received")
	}

	indices := make([]int, 0, len(partialCalls))
	for index := range partialCalls {
		indices = append(indices, index)
	}
	sort.Ints(indices)
	toolCalls := make([]chatToolCall, 0, len(indices))
	for _, index := range indices {
		call := *partialCalls[index]
		if call.ID == "" {
			call.ID = fmt.Sprintf("call_%d", index)
		}
		toolCalls = append(toolCalls, call)
	}

	return chatResponse{
		Message: chatMessage{
			Role:      "assistant",
			Content:   content.String(),
			ToolCalls: toolCalls,
		},
		TextStreamed: streamedText,
	}, nil
}

func scanSSEData(r io.Reader, handle func(string) error) error {
	scanner := bufio.NewScanner(r)
	scanner.Buffer(make([]byte, 64<<10), 16<<20)
	var dataLines []string

	dispatch := func() error {
		if len(dataLines) == 0 {
			return nil
		}
		data := strings.Join(dataLines, "\n")
		dataLines = dataLines[:0]
		return handle(data)
	}

	for scanner.Scan() {
		line := scanner.Text()
		if line == "" {
			if err := dispatch(); err != nil {
				return err
			}
			continue
		}
		if strings.HasPrefix(line, "data:") {
			value := strings.TrimPrefix(line, "data:")
			value = strings.TrimPrefix(value, " ")
			dataLines = append(dataLines, value)
		}
	}
	if err := dispatch(); err != nil {
		return err
	}
	if err := scanner.Err(); err != nil {
		return fmt.Errorf("read OpenAI-compatible SSE response: %w", err)
	}
	return nil
}
