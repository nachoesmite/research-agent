// Research Agent Schema Definitions
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

// Functions for Research Agent

function CreateAnalysts(topic: string, human_analyst_feedback: string, max_analysts: int) -> Perspectives {
  client GPT4o
  prompt #"
    You are tasked with creating a set of AI analyst personas. Follow these instructions carefully:

    1. First, review the research topic:
    {{ topic }}
            
    2. Examine any editorial feedback that has been optionally provided to guide creation of the analysts: 
    {{ human_analyst_feedback }}
        
    3. Determine the most interesting themes based upon documents and/or feedback above.
                        
    4. Pick the top {{ max_analysts }} themes.

    5. Assign one analyst to each theme.

    Generate the set of analysts with their roles, affiliations, and descriptions.

    {{ ctx.output_format }}
  "#
}

test create_analysts_test() {
  functions [CreateAnalysts]
  args {
    topic #"
      best player of peñarol club
    "#
    human_analyst_feedback ""
    max_analysts 1
  }
  @@check(correct_count, {{ this.analysts|length == 1 }})
  @@check(has_name, {{ this.analysts[0].name|length > 0 }})
  @@check(has_role, {{ this.analysts[0].role|length > 0 }})
  @@check(has_affiliation, {{ this.analysts[0].affiliation|length > 0 }})
  @@check(has_description, {{ this.analysts[0].description|length > 0 }})
  @@assert({{ _.checks.correct_count and _.checks.has_name and _.checks.has_role }})
}

function GenerateQuestion(analyst_persona: string, messages: Message[]) -> string {
  client GPT4o
  prompt #"
    You are an analyst tasked with interviewing an expert to learn about a specific topic. 

    Your goal is boil down to interesting and specific insights related to your topic.

    1. Interesting: Insights that people will find surprising or non-obvious.
            
    2. Specific: Insights that avoid generalities and include specific examples from the expert.

    Here is your persona and goals: {{ analyst_persona }}

    Previous conversation:
    {% for message in messages %}
    {{ message.role }}: {{ message.content }}
    {% endfor %}
            
    Begin by introducing yourself using a name that fits your persona, and then ask your question.

    Continue to ask questions to drill down and refine your understanding of the topic.
            
    When you are satisfied with your understanding, complete the interview with: "Thank you so much for your help!"

    Remember to stay in character throughout your response, reflecting the persona and goals provided to you.

    {{ ctx.output_format }}
  "#
}

test generate_question_test() {
  functions [GenerateQuestion]
  args {
    analyst_persona #"
      You are Dr. Sarah Chen, a sports historian and analyst specializing in South American football history. 
      Your affiliation is with the Institute of Football Culture at Universidad de la República, Uruguay. 
      You focus on legendary players and their impact on club culture, particularly interested in technical skills, 
      leadership qualities, and cultural significance within club traditions.
    "#
    messages [
      {
        role: "system"
        content: "You are an expert on Peñarol football club history and players."
      },
      {
        role: "assistant"
        content: "Hello, I'm Dr. Sarah Chen from the Institute of Football Culture. I'm researching the greatest players in Peñarol's history."
      }
    ]
  }
  @@check(substantial, {{ this|length > 20 }})
  @@check(has_question_mark, {{ "?" in this }})
  @@check(introduces_herself, {{ "Sarah" in this or "Dr." in this }})
  @@check(not_ending_interview, {{ "Thank you so much for your help!" not in this }})
  @@assert({{ _.checks.substantial and _.checks.has_question_mark }})
}

function GenerateSearchQuery(messages: Message[]) -> SearchQuery {
  client GPT4o
  prompt #"
    You will be given a conversation between an analyst and an expert. 

    Your goal is to generate a well-structured query for use in retrieval and/or web-search related to the conversation.
            
    First, analyze the full conversation.

    Pay particular attention to the final question posed by the analyst.

    Convert this final question into a well-structured web search query.

    Conversation:
    {% for message in messages %}
    {{ message.role }}: {{ message.content }}
    {% endfor %}

    {{ ctx.output_format }}
  "#
}

test generate_search_query_test() {
  functions [GenerateSearchQuery]
  args {
    messages [
      {
        role: "assistant"
        content: "I'm researching the greatest Peñarol players of all time."
      },
      {
        role: "user"
        content: "Many consider Diego Forlán to be one of the most talented players to come from Peñarol's youth system."
      },
      {
        role: "assistant"
        content: "Can you tell me more about Diego Forlán's early career at Peñarol and what made him stand out from other young talents?"
      }
    ]
  }
  @@check(has_query, {{ this.search_query|length > 0 }})
  @@check(substantial_query, {{ this.search_query|length > 10 }})
  @@check(includes_forlan, {{ "Forlán" in this.search_query or "Forlan" in this.search_query }})
  @@check(includes_penarol, {{ "Peñarol" in this.search_query or "Penarol" in this.search_query }})
  @@assert({{ _.checks.has_query and _.checks.substantial_query }})
}

function GenerateAnswer(analyst_persona: string, context: string, messages: Message[]) -> string {
  client GPT4o
  prompt #"
    You are an expert being interviewed by an analyst.

    Here is analyst area of focus: {{ analyst_persona }}
            
    You goal is to answer a question posed by the interviewer.

    ---------------------------------
    The question is taken from: 
    {% for message in messages %}
      {{ message.role }}: {{ message.content }}
    {% endfor %}
    ---------------------------------
    To answer question, use this context:
    {{ context }}

    When answering questions, follow these guidelines:
            
    1. Use only the information provided in the context. 
            
    2. Do not introduce external information or make assumptions beyond what is explicitly stated in the context.

    3. The context contain sources at the topic of each individual document.

    4. Include these sources your answer next to any relevant statements. For example, for source # 1 use [1]. 

    5. List your sources in order at the bottom of your answer. [1] Source 1, [2] Source 2, etc
            
    6. If the source is: <Document source="assistant/docs/llama3_1.pdf" page="7"/>' then just list: 
            
    [1] assistant/docs/llama3_1.pdf, page 7 

    7. Do not forget wikipedia links!
            
    And skip the addition of the brackets as well as the Document source preamble in your citation.

    {{ ctx.output_format }}
  "#
}

test generate_answer_test() {
  functions [GenerateAnswer]
  args {
    analyst_persona #"
      Sports historian focusing on South American football legends, particularly interested in 
      technical skills and cultural impact of players within their clubs.
    "#
    context #"
      <Document source="peñarol_history.pdf" page="15">
      Diego Forlán joined Peñarol's youth academy at age 12 in 1991. He quickly distinguished himself 
      with exceptional ball control and vision. His grandfather, Juan Carlos Corazzo, was also a former 
      Peñarol player, creating a family legacy. Forlán made his first-team debut in 1997 at age 18, 
      scoring 11 goals in his first season.
      </Document>
      
      <Document source="https://wikipedia.org/wiki/Diego_Forlan">
      Diego Forlán Corazzo is a Uruguayan former professional footballer who played as a forward. 
      He is regarded as one of the greatest Uruguayan players of all time. Forlán had a successful 
      international career, winning the Golden Ball at the 2010 FIFA World Cup.
      </Document>
    "#
    messages [
      {
        role: "assistant"
        content: "I'm researching the greatest Peñarol players of all time."
      },
      {
        role: "user"
        content: "Many consider Diego Forlán to be one of the most talented players to come from Peñarol's youth system."
      },
      {
        role: "assistant"
        content: "Can you tell me more about Diego Forlán's early career at Peñarol and what made him stand out from other young talents?"
      }
    ]
  }
  @@check(substantial, {{ this|length > 100 }})
  @@check(has_citations, {{ "[1]" in this or "[2]" in this }})
  @@check(mentions_forlan, {{ "Diego Forlán" in this or "Forlan" in this }})
  @@check(has_sources_section, {{ "peñarol_history.pdf" in this and "wikipedia.org" in this }})
  @@assert({{ _.checks.substantial and _.checks.has_citations }})
}

function WriteSection(analyst_description: string, context: string) -> ReportSection {
  client GPT4o
  prompt #"
    You are an expert technical writer. 
                
    Your task is to create a short, easily digestible section of a report based on a set of source documents.

    1. Analyze the content of the source documents: 
    - The name of each source document is at the start of the document, with the <Document tag.
            
    2. Create a report structure using markdown formatting:
    - Use ## for the section title
    - Use ### for sub-section headers
            
    3. Write the report following this structure:
    a. Title (## header)
    b. Summary (### header)
    c. Sources (### header)

    4. Make your title engaging based upon the focus area of the analyst: 
    {{ analyst_description }}

    5. For the summary section:
    - Set up summary with general background / context related to the focus area of the analyst
    - Emphasize what is novel, interesting, or surprising about insights gathered from the interview
    - Create a numbered list of source documents, as you use them
    - Do not mention the names of interviewers or experts
    - Aim for approximately 400 words maximum
    - Use numbered sources in your report (e.g., [1], [2]) based on information from source documents
            
    6. In the Sources section:
    - Include all sources used in your report
    - Provide full links to relevant websites or specific document paths
    - Separate each source by a newline. Use two spaces at the end of each line to create a newline in Markdown.
    - It will look like:

    ### Sources
    [1] Link or Document name
    [2] Link or Document name

    7. Be sure to combine sources. For example this is not correct:

    [3] https://ai.meta.com/blog/meta-llama-3-1/
    [4] https://ai.meta.com/blog/meta-llama-3-1/

    There should be no redundant sources. It should simply be:

    [3] https://ai.meta.com/blog/meta-llama-3-1/
            
    8. Final review:
    - Ensure the report follows the required structure
    - Include no preamble before the title of the report
    - Check that all guidelines have been followed

    Source material:
    {{ context }}

    {{ ctx.output_format }}
  "#
}

test write_section_test() {
  functions [WriteSection]
  args {
    analyst_description #"
      Sports historian and cultural analyst focusing on legendary South American football players, 
      their technical abilities, leadership qualities, and lasting impact on club traditions and culture.
    "#
    context #"
      <Document source="peñarol_legends.pdf" page="23">
      Diego Forlán's time at Peñarol (1997-2001) established him as one of the most promising talents 
      in Uruguayan football. His 11 goals in his debut season broke the club's record for a teenage player. 
      His technical skills, particularly his first touch and ability to create space, were exceptional for his age.
      </Document>
      
      <Document source="https://en.wikipedia.org/wiki/Diego_Forlan">
      Forlán won the Pichichi Trophy twice in La Liga and was the top scorer at the 2010 World Cup. 
      His success internationally brought global recognition to Peñarol's youth development system.
      </Document>
      
      <Document source="club_culture_study.pdf" page="41">
      Former teammates describe Forlán as a natural leader despite his young age, often staying after 
      training to work on technique. His dedication became legendary among the youth academy players 
      who followed his example.
      </Document>
    "#
  }
  @@check(has_main_header, {{ "##" in this.content }})
  @@check(has_sub_headers, {{ "###" in this.content }})
  @@check(has_summary, {{ "### Summary" in this.content }})
  @@check(has_sources, {{ "### Sources" in this.content }})
  @@check(has_citations, {{ "[1]" in this.content }})
  @@check(reasonable_length, {{ this.content|length < 10000 }})
  @@assert({{ _.checks.has_main_header and _.checks.has_summary and _.checks.has_sources }})
}

function WriteReport(topic: string, sections: string) -> string {
  client GPT4o
  prompt #"
    You are a technical writer creating a report on this overall topic: 
    {{ topic }}
        
    You have a team of analysts. Each analyst has done two things: 

    1. They conducted an interview with an expert on a specific sub-topic.
    2. They write up their finding into a memo.

    Your task: 

    1. You will be given a collection of memos from your analysts.
    2. Think carefully about the insights from each memo.
    3. Consolidate these into a crisp overall summary that ties together the central ideas from all of the memos. 
    4. Summarize the central points in each memo into a cohesive single narrative.

    To format your report:
     
    1. Use markdown formatting. 
    2. Include no pre-amble for the report.
    3. Use no sub-heading. 
    4. Start your report with a single title header: ## Insights
    5. Do not mention any analyst names in your report.
    6. Preserve any citations in the memos, which will be annotated in brackets, for example [1] or [2].
    7. Create a final, consolidated list of sources and add to a Sources section with the `## Sources` header.
    8. List your sources in order and do not repeat.

    [1] Source 1
    [2] Source 2

    Here are the memos from your analysts to build your report from: 
    {{ sections }}

    {{ ctx.output_format }}
  "#
}

test write_report_test() {
  functions [WriteReport]
  args {
    topic "Greatest Players in Peñarol Football Club History"
    sections #"
      ## Technical Excellence and Youth Development
      
      ### Summary
      Diego Forlán's emergence from Peñarol's youth academy represents one of the most successful 
      talent development stories in Uruguayan football [1]. His record-breaking debut season with 
      11 goals established new benchmarks for teenage players [1]. The technical foundation he 
      developed at Peñarol became the cornerstone of his later international success [2].
      
      ### Sources
      [1] peñarol_legends.pdf, page 23  
      [2] https://en.wikipedia.org/wiki/Diego_Forlan
      
      ## Cultural Impact and Leadership
      
      ### Summary
      Beyond technical skills, Forlán's influence on club culture was profound [1]. His dedication 
      to extra training sessions inspired younger players and established new standards within the 
      academy [1]. This cultural legacy continues to influence Peñarol's player development approach.
      
      ### Sources
      [1] club_culture_study.pdf, page 41
    "#
  }
  @@check(starts_with_insights, {{ "## Insights" in this }})
  @@check(has_sources_section, {{ "## Sources" in this }})
  @@check(preserves_citations, {{ "[1]" in this }})
  @@check(no_analyst_names, {{ "Dr." not in this and "Sarah" not in this }})
  @@check(consolidated_content, {{ this|length > 100 }})
  @@assert({{ _.checks.starts_with_insights and _.checks.has_sources_section }})
}

function WriteIntroduction(topic: string, sections: string) -> string {
  client GPT4o
  prompt #"
    You are a technical writer finishing a report on {{ topic }}

    You will be given all of the sections of the report.

    You job is to write a crisp and compelling introduction section.

    Include no pre-amble for the section.

    Target around 100 words, crisply previewing all of the sections of the report.

    Use markdown formatting. 

    Create a compelling title and use the # header for the title.

    Use ## Introduction as the section header. 

    Here are the sections to reflect on for writing: 
    {{ sections }}

    {{ ctx.output_format }}
  "#
}

test write_introduction_test() {
  functions [WriteIntroduction]
  args {
    topic "Greatest Players in Peñarol Football Club History"
    sections #"
      ## Technical Excellence and Youth Development
      Diego Forlán's emergence from Peñarol's youth academy represents a masterclass in talent development, 
      setting records and establishing new benchmarks for future generations.
      
      ## Cultural Impact and Leadership  
      Beyond individual achievements, these legendary players shaped the very culture of the club, 
      creating lasting traditions that continue to influence Peñarol today.
      
      ## International Recognition and Legacy
      The global success of Peñarol-developed players has elevated the club's reputation worldwide, 
      attracting international attention to Uruguay's football heritage.
    "#
  }
  @@check(has_main_title, {{ "# " in this }})
  @@check(has_introduction_header, {{ "## Introduction" in this }})
  @@check(appropriate_length, {{ this|length >= 50 and this|length <= 1000 }})
  @@check(no_preamble, {{ not (this[:7] == "Here is" or this[:7] == "This is") }})
  @@assert({{ _.checks.has_main_title and _.checks.has_introduction_header }})
}

function WriteConclusion(topic: string, sections: string) -> string {
  client GPT4o
  prompt #"
    You are a technical writer finishing a report on {{ topic }}

    You will be given all of the sections of the report.

    You job is to write a crisp and compelling conclusion section.

    Include no pre-amble for the section.

    Target around 100 words, crisply recapping all of the sections of the report.

    Use markdown formatting. 

    Use ## Conclusion as the section header.

    Here are the sections to reflect on for writing: 
    {{ sections }}

    {{ ctx.output_format }}
  "#
}

test write_conclusion_test() {
  functions [WriteConclusion]
  args {
    topic "Greatest Players in Peñarol Football Club History"
    sections #"
      ## Technical Excellence and Youth Development
      The analysis reveals how Peñarol's systematic approach to youth development has consistently 
      produced world-class talent, with players like Diego Forlán serving as prime examples of 
      the club's technical excellence standards.
      
      ## Cultural Impact and Leadership
      These legendary figures didn't just excel individually but fundamentally shaped the club's 
      identity, creating a culture of dedication and excellence that continues to inspire new generations.
      
      ## International Recognition and Legacy
      The global achievements of Peñarol-developed players have established the club as a prestigious 
      institution in world football, cementing its place in international football history.
    "#
  }
  @@check(has_conclusion_header, {{ "## Conclusion" in this }})
  @@check(appropriate_length, {{ this|length >= 50 and this|length <= 1000 }})
  @@check(no_preamble, {{ not (this[:7] == "Here is" or this[:7] == "This is") }})
  @@check(recaps_content, {{ "Peñarol" in this }})
  @@assert({{ _.checks.has_conclusion_header and _.checks.appropriate_length }})
}