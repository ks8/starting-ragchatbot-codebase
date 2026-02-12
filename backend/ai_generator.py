import anthropic
from typing import List, Optional, Dict, Any

class AIGenerator:
    """Handles interactions with Anthropic's Claude API for generating responses"""
    
    # Static system prompt to avoid rebuilding on each call
    SYSTEM_PROMPT = """ You are an AI assistant specialized in course materials and educational content with access to search tools for course information.

Search Tool Usage:
- Use the search_course_content tool **only** for questions about specific course content or detailed educational materials
- Use the get_course_outline tool for questions about a course's structure, outline, lesson list, or what topics a course covers
- You may use up to two tools sequentially per query when needed (e.g., get a course outline first, then search for specific content)
- Each tool call is a separate step — review results before deciding if another call is needed
- Synthesize tool results into accurate, fact-based responses
- If a tool yields no results, state this clearly without offering alternatives

Outline Queries:
- When the user asks about a course outline, structure, or lesson list, use the get_course_outline tool
- Always include in your response: the course title, the course link, and every lesson with its number and title

Response Protocol:
- **General knowledge questions**: Answer using existing knowledge without searching
- **Course-specific questions**: Use the appropriate tool first, then answer
- **No meta-commentary**:
 - Provide direct answers only — no reasoning process, search explanations, or question-type analysis
 - Do not mention "based on the search results"


All responses must be:
1. **Brief, Concise and focused** - Get to the point quickly
2. **Educational** - Maintain instructional value
3. **Clear** - Use accessible language
4. **Example-supported** - Include relevant examples when they aid understanding
Provide only the direct answer to what was asked.
"""

    MAX_TOOL_ROUNDS = 2

    def __init__(self, api_key: str, model: str):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model
        
        # Pre-build base API parameters
        self.base_params = {
            "model": self.model,
            "temperature": 0,
            "max_tokens": 800
        }
    
    def generate_response(self, query: str,
                         conversation_history: Optional[str] = None,
                         tools: Optional[List] = None,
                         tool_manager=None) -> str:
        """
        Generate AI response with optional tool usage and conversation context.
        
        Args:
            query: The user's question or request
            conversation_history: Previous messages for context
            tools: Available tools the AI can use
            tool_manager: Manager to execute tools
            
        Returns:
            Generated response as string
        """
        
        # Build system content efficiently - avoid string ops when possible
        system_content = (
            f"{self.SYSTEM_PROMPT}\n\nPrevious conversation:\n{conversation_history}"
            if conversation_history 
            else self.SYSTEM_PROMPT
        )
        
        # Prepare API call parameters efficiently
        api_params = {
            **self.base_params,
            "messages": [{"role": "user", "content": query}],
            "system": system_content
        }
        
        # Add tools if available
        if tools:
            api_params["tools"] = tools
            api_params["tool_choice"] = {"type": "auto"}
        
        # Get response from Claude
        response = self.client.messages.create(**api_params)
        
        # Handle tool execution if needed
        if response.stop_reason == "tool_use" and tool_manager:
            return self._handle_tool_execution(response, api_params, tool_manager)
        
        # Return direct response
        return response.content[0].text
    
    def _handle_tool_execution(self, initial_response, base_params: Dict[str, Any], tool_manager):
        """
        Handle execution of tool calls across multiple rounds.

        Supports up to MAX_TOOL_ROUNDS sequential tool-call rounds, enabling
        multi-step reasoning (e.g., get outline then search content).

        Args:
            initial_response: The response containing tool use requests
            base_params: Base API parameters
            tool_manager: Manager to execute tools

        Returns:
            Final response text after tool execution
        """
        messages = base_params["messages"].copy()
        current_response = initial_response

        for round_num in range(self.MAX_TOOL_ROUNDS):
            # 1. Append assistant's tool-use response
            messages.append({"role": "assistant", "content": current_response.content})

            # 2. Execute tool calls, collect results
            tool_results = []
            tool_error = False
            for block in current_response.content:
                if block.type == "tool_use":
                    try:
                        result = tool_manager.execute_tool(block.name, **block.input)
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": result
                        })
                    except Exception as e:
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": str(e),
                            "is_error": True
                        })
                        tool_error = True

            # 3. Append tool results
            if tool_results:
                messages.append({"role": "user", "content": tool_results})

            # 4. Determine if this is the last round
            is_last_round = (round_num == self.MAX_TOOL_ROUNDS - 1) or tool_error

            # 5. Build follow-up API call
            follow_up_params = {
                **self.base_params,
                "messages": messages,
                "system": base_params["system"]
            }
            if not is_last_round:
                follow_up_params["tools"] = base_params["tools"]
                follow_up_params["tool_choice"] = {"type": "auto"}

            # 6. Call the API
            current_response = self.client.messages.create(**follow_up_params)

            # 7. If Claude didn't request another tool, return text
            if current_response.stop_reason != "tool_use":
                return current_response.content[0].text

        # Exhausted all rounds — return whatever text we got
        return current_response.content[0].text