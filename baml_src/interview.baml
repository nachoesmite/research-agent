// Interview Graph Functions

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