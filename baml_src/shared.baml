// Shared Schema Definitions
client<llm> GPT4o {
  provider "openai"
  options {
    model "gpt-4o"
    // api_key defaults to env.OPENAI_API_KEY
  }
}

class Analyst {
  affiliation string @description("Primary affiliation of the analyst")
  name string @description("Name of the analyst")
  role string @description("Role of the analyst in the context of the topic")
  description string @description("Description of the analyst focus, concerns, and motives")
}

class Perspectives {
  analysts Analyst[] @description("Comprehensive list of analysts with their roles and affiliations")
}

class SearchQuery {
  search_query string @description("Search query for retrieval")
}

class Message {
  role string @description("user, assistant, or system")
  content string @description("The message content")
  name string? @description("Optional name of the message sender")
}

class InterviewResponse {
  response string @description("The response content")
  should_continue bool @description("Whether the interview should continue")
}

class ReportSection {
  content string @description("The section content in markdown format")
}