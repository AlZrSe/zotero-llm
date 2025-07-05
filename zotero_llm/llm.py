def analyze_papers(zot, query, credentials):
    """Analyze papers from local Zotero library using LLM"""
    try:
        items = zot.items(limit=5, sort='dateModified', direction='desc')
    except Exception as e:
        console.print(f"[red]Failed to connect to local Zotero: {e}")
        return "Could not fetch items from local Zotero."

    # Prepare context for LLM
    papers_context = []
    for item in items:
        data = item.get('data', item)
        if data.get('title'):
            context = {
                'title': data.get('title', ''),
                'abstract': data.get('abstractNote', ''),
                'authors': data.get('creators', []),
                'year': data.get('date', '')
            }
            papers_context.append(context)
    
    # Prepare prompt for LLM
    system_prompt = """You are a research assistant analyzing academic papers.
    Based on the provided papers and query, provide insights and suggestions."""
    
    user_prompt = f"""Query: {query}
    
    Papers:
    {papers_context}
    
    Please provide:
    1. Key insights related to the query
    2. Connections between papers
    3. Suggestions for further research"""
    
    # Call OpenAI API
    client = openai.OpenAI(api_key=credentials['openai_api_key'],
                           base_url=credentials['llm_base_url'])
    response = client.chat.completions.create(
        model=credentials['llm_model'],
        extra_query={"provider": "OpenaiChat"},
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    )
    
    return response.choices[0].message.content

def ask_llm(query, zotero_list, llm_client, credentials):
    """Ask the LLM a question with context and return the response"""

    context = "\n\n".join([f'Title: {item.get("title", "")}\nAbstract: {item.get("abstract", "")}\nYear: {item.get("year", "")}' for item in zotero_list])

     # Prepare prompt for LLM
    system_prompt = """You are a research assistant analyzing academic papers.
    Based on the provided papers and query, provide useful thoughts, summary, insights and suggestions.
    Also, provide a citations as number in square brackets with a reference list of the papers used at the end of your response."""

    user_prompt = f"""Query: {query}
    
    Papers:
    {context}"""

    response = llm_client.chat.completions.create(
        model='auto', # credentials['llm_model'],
        extra_query={"provider": "OpenaiChat"},
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    )
    return response.choices[0].message.content