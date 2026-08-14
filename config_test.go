package main

import "testing"

func TestChatCompletionsEndpoint(t *testing.T) {
	tests := map[string]string{
		"https://api.example.com":                         "https://api.example.com/v1/chat/completions",
		"https://api.example.com/":                        "https://api.example.com/v1/chat/completions",
		"https://api.example.com/v1":                      "https://api.example.com/v1/chat/completions",
		"https://api.example.com/v1/":                     "https://api.example.com/v1/chat/completions",
		"https://api.example.com/v1/chat/completions":     "https://api.example.com/v1/chat/completions",
		"http://localhost:8080/openai/v1":                 "http://localhost:8080/openai/v1/chat/completions",
		"http://localhost:8080/gateway":                   "http://localhost:8080/gateway/v1/chat/completions",
		"http://localhost:8080/gateway/chat/completions/": "http://localhost:8080/gateway/chat/completions",
	}

	for input, expected := range tests {
		t.Run(input, func(t *testing.T) {
			actual, err := chatCompletionsEndpoint(input)
			if err != nil {
				t.Fatalf("chatCompletionsEndpoint() error = %v", err)
			}
			if actual != expected {
				t.Fatalf("chatCompletionsEndpoint() = %q, want %q", actual, expected)
			}
		})
	}
}

func TestValidateEndpointUsesExactURL(t *testing.T) {
	input := "https://gateway.example.com/custom/generate"
	actual, err := validateEndpoint(input)
	if err != nil {
		t.Fatalf("validateEndpoint() error = %v", err)
	}
	if actual != input {
		t.Fatalf("validateEndpoint() = %q, want exact %q", actual, input)
	}
}

func TestLoadConfigPrefersExactEndpoint(t *testing.T) {
	t.Setenv(envAPIKey, "key")
	t.Setenv(envEndpoint, "https://gateway.example.com/custom/generate")
	t.Setenv(envBaseURL, "https://ignored.example.com/v1")
	t.Setenv(envModel, "model")

	config, err := loadConfig()
	if err != nil {
		t.Fatalf("loadConfig() error = %v", err)
	}
	if config.Endpoint != "https://gateway.example.com/custom/generate" {
		t.Fatalf("Endpoint = %q", config.Endpoint)
	}
}

func TestEndpointRejectsInvalidURL(t *testing.T) {
	for _, input := range []string{"", "api.example.com", "ftp://api.example.com", "https://api.example.com?v=1"} {
		if _, err := chatCompletionsEndpoint(input); err == nil {
			t.Errorf("chatCompletionsEndpoint(%q) unexpectedly succeeded", input)
		}
	}
}
