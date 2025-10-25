// Researcher Graph Functions

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