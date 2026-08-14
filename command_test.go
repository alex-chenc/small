package main

import (
	"bytes"
	"context"
	"strings"
	"testing"
	"time"
)

func TestCommandExecutorExitCodeAndSecretFiltering(t *testing.T) {
	t.Setenv(envAPIKey, "must-not-reach-child")
	t.Setenv(envEndpoint, "https://endpoint-must-not-reach-child")
	t.Setenv(envBaseURL, "https://example.test/v1")
	t.Setenv(envModel, "test-model")

	var terminalOut, terminalErr bytes.Buffer
	executor := CommandExecutor{cwd: t.TempDir()}
	result := executor.Execute(
		context.Background(),
		`printf '%s|%s|%s|%s' "$OPENAI_API_KEY" "$OPENAI_ENDPOINT" "$OPENAI_BASE_URL" "$OPENAI_MODEL"; printf 'problem' >&2; exit 7`,
		time.Second,
		&terminalOut,
		&terminalErr,
	)

	if result.ExitCode != 7 {
		t.Fatalf("ExitCode = %d, want 7", result.ExitCode)
	}
	if result.Stdout != "|||" || terminalOut.String() != "|||" {
		t.Fatalf("OpenAI environment reached child process: captured=%q terminal=%q", result.Stdout, terminalOut.String())
	}
	if result.Stderr != "problem" || terminalErr.String() != "problem" {
		t.Fatalf("stderr = %q, terminal stderr = %q", result.Stderr, terminalErr.String())
	}
}

func TestCommandExecutorTimeout(t *testing.T) {
	executor := CommandExecutor{cwd: t.TempDir()}
	started := time.Now()
	result := executor.Execute(context.Background(), "sleep 5", 50*time.Millisecond, &bytes.Buffer{}, &bytes.Buffer{})

	if !result.TimedOut || result.ExitCode != 124 {
		t.Fatalf("result = %+v, want timed out with exit code 124", result)
	}
	if elapsed := time.Since(started); elapsed > 2*time.Second {
		t.Fatalf("timed out command took too long: %s", elapsed)
	}
}

func TestCappedBufferKeepsHeadAndTail(t *testing.T) {
	buffer := newCappedBuffer(10)
	_, _ = buffer.Write([]byte("abcdef"))
	_, _ = buffer.Write([]byte("ghijklmnop"))

	if !buffer.Truncated() {
		t.Fatal("buffer was not marked truncated")
	}
	result := buffer.String()
	if !strings.HasPrefix(result, "abcde") || !strings.HasSuffix(result, "lmnop") {
		t.Fatalf("buffer = %q, want first and last five bytes", result)
	}
}
