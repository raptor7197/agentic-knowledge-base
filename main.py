import os
import google.generativeai as genai
from dotenv import load_dotenv
from tools import (
    read_file, search_code, list_directory, run_command, change_directory,
    add_to_vectorstore, search_vectorstore, index_codebase
)

load_dotenv()

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

read_file_func = genai.protos.FunctionDeclaration(
    name="read_file",
    description="Reads the contents of a file. The path can be relative to the current directory or absolute.",
    parameters=genai.protos.Schema(
        type=genai.protos.Type.OBJECT,
        properties={
            "file_path": genai.protos.Schema(
                type=genai.protos.Type.STRING,
                description="The path to the file to read (relative or absolute)."
            )
        },
        required=["file_path"]
    )
)

search_code_func = genai.protos.FunctionDeclaration(
    name="search_code",
    description="Searches for a pattern in code files within a directory. The path can be relative or absolute.",
    parameters=genai.protos.Schema(
        type=genai.protos.Type.OBJECT,
        properties={
            "pattern": genai.protos.Schema(
                type=genai.protos.Type.STRING,
                description="The search pattern or text to find."
            ),
            "path": genai.protos.Schema(
                type=genai.protos.Type.STRING,
                description="The directory path to search in (default: current directory)."
            )
        },
        required=["pattern"]
    )
)

list_directory_func = genai.protos.FunctionDeclaration(
    name="list_directory",
    description="Lists all files and directories in the specified path. The path can be relative or absolute.",
    parameters=genai.protos.Schema(
        type=genai.protos.Type.OBJECT,
        properties={
            "path": genai.protos.Schema(
                type=genai.protos.Type.STRING,
                description="The directory path to list (default: current directory)."
            )
        }
    )
)

run_command_func = genai.protos.FunctionDeclaration(
    name="run_command",
    description="Executes a shell command from the agent's current working directory.",
    parameters=genai.protos.Schema(
        type=genai.protos.Type.OBJECT,
        properties={
            "command": genai.protos.Schema(
                type=genai.protos.Type.STRING,
                description="The shell command to execute."
            )
        },
        required=["command"]
    )
)

change_directory_func = genai.protos.FunctionDeclaration(
    name="change_directory",
    description="Changes the agent's current working directory. All subsequent file operations will be relative to this new directory.",
    parameters=genai.protos.Schema(
        type=genai.protos.Type.OBJECT,
        properties={
            "path": genai.protos.Schema(
                type=genai.protos.Type.STRING,
                description="The path to the new directory (relative or absolute)."
            )
        },
        required=["path"]
    )
)

search_vectorstore_func = genai.protos.FunctionDeclaration(
    name="search_vectorstore",
    description="Search the vector database for semantically similar code or content.",
    parameters=genai.protos.Schema(
        type=genai.protos.Type.OBJECT,
        properties={
            "query": genai.protos.Schema(
                type=genai.protos.Type.STRING,
                description="The search query to find similar content."
            ),
            "k": genai.protos.Schema(
                type=genai.protos.Type.INTEGER,
                description="Number of results to return (default: 5)."
            )
        },
        required=["query"]
    )
)

add_to_vectorstore_func = genai.protos.FunctionDeclaration(
    name="add_to_vectorstore",
    description="Add a file's content to the vector database. The path can be relative or absolute.",
    parameters=genai.protos.Schema(
        type=genai.protos.Type.OBJECT,
        properties={
            "file_path": genai.protos.Schema(
                type=genai.protos.Type.STRING,
                description="The path to the file to add to the vector database."
            )
        },
        required=["file_path"]
    )
)

index_codebase_func = genai.protos.FunctionDeclaration(
    name="index_codebase",
    description="Index all code files in a directory for semantic search. The path can be relative or absolute.",
    parameters=genai.protos.Schema(
        type=genai.protos.Type.OBJECT,
        properties={
            "directory_path": genai.protos.Schema(
                type=genai.protos.Type.STRING,
                description="The directory path to index (default: current directory)."
            )
        }
    )
)

# Create tool with all function declarations
tools = genai.protos.Tool(
    function_declarations=[
        read_file_func,
        search_code_func,
        list_directory_func,
        run_command_func,
        change_directory_func,
        search_vectorstore_func,
        add_to_vectorstore_func,
        index_codebase_func
    ]
)

# Create model with tools
model = genai.GenerativeModel(
    'gemini-2.0-flash-exp',
    tools=[tools]
)

# Start chat session
chat = model.start_chat()

# Tool function mapping
tool_functions = {
    'read_file': read_file,
    'search_code': search_code,
    'list_directory': list_directory,
    'run_command': run_command,
    'change_directory': change_directory,
    'search_vectorstore': search_vectorstore,
    'add_to_vectorstore': add_to_vectorstore,
    'index_codebase': index_codebase
}

# Interactive loop
while True:
    user_input = input("\nInput (quit to exit) ")
    if user_input.lower() == 'quit':
        break

    prompt = f"""You are an expert coding assistant for professional development environments.
Analyze the codebase and provide help with the user's coding task: {user_input}.
Use available tools to read files, search code, list directories, and run commands.
Provide detailed analysis, suggestions for optimization, and specific code changes if needed.
"""

    # Send message and handle function calling loop
    response = chat.send_message(prompt)
    
    # Maximum iterations to prevent infinite loops
    max_iterations = 10
    iteration = 0
    
    while iteration < max_iterations:
        # Check if response contains function calls
        parts = response.candidates[0].content.parts
        function_calls = [part for part in parts if part.function_call.name]
        
        if not function_calls:
            # No more function calls, print final response
            print("\n" + response.text)
            break
        
        # Execute each function call
        function_responses = []
        for part in function_calls:
            function_call = part.function_call
            function_name = function_call.name
            function_args = dict(function_call.args)
            
            print(f"\n[Calling tool: {function_name} with args: {function_args}]")
            
            # Execute the function
            try:
                if function_name in tool_functions:
                    result = tool_functions[function_name](**function_args)
                else:
                    result = f"Error: Unknown function {function_name}"
            except Exception as e:
                result = f"Error executing {function_name}: {str(e)}"
            
            # Create function response
            function_responses.append(
                genai.protos.Part(
                    function_response=genai.protos.FunctionResponse(
                        name=function_name,
                        response={"result": str(result)}
                    )
                )
            )
        
        # Send function responses back to the model
        response = chat.send_message(function_responses)
        iteration += 1
    
    if iteration >= max_iterations:
        print("\n[Warning: Maximum function call iterations reached]")
    
    # Optional feedback collection
    feedback = input("\nWas this response helpful? (yes/no): ")
    if feedback.lower() == 'no':
        refinement = input("How can I improve? ")
        print("Feedback noted for self-optimization.")
